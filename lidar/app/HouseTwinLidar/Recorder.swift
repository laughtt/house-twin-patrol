import ARKit
import CoreImage
import CoreImage.CIFilterBuiltins
import RealityKit

/// Records RGB + LiDAR depth + ARKit pose at ~4 fps into Documents/<session>/ and packs it as a zip.
/// Pose convention: `T` is ARKit's camera-to-world 4x4, row-major. ARKit world: metres, y up (gravity),
/// camera axes: x right, y up, z backwards (the camera looks along -z), in the sensor's landscape image.
final class Recorder: NSObject, ObservableObject, ARSessionDelegate {
    static let fps = 4.0, width = 960.0
    let supported = ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
    let view = ARView(frame: .zero, cameraMode: .ar, automaticallyConfigureSession: false)
    @Published var recording = false
    @Published var frames = 0
    @Published var bytes = 0
    @Published var tracking = "starting"
    @Published var room = "Hallway"
    @Published var status = ""
    @Published var zip: URL?
    @Published var sessionID = ""
    @Published var reviewing = false
    @Published var area = 0.0
    @Published var growth = 0.0
    let veil = AnchorEntity(.camera)
    private(set) var dir: URL!
    private let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    private let queue = DispatchQueue(label: "writer")
    private let meshQueue = DispatchQueue(label: "mesh", qos: .utility)
    private let ci = CIContext()
    private var lines: FileHandle?
    private var events: FileHandle?
    private var t0: TimeInterval = 0, last: TimeInterval = 0, now: TimeInterval = 0, lastMesh: TimeInterval = 0
    private var pose: [Float] = []
    private var busy = false
    private var progress: NSKeyValueObservation?

    func run() {
        guard supported else { return }
        view.session.delegate = self
        view.session.run(config)
        guard ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) else { return }
        // scanned surfaces (the ARKit mesh) occlude a dark plane 5 m ahead, so only un-scanned areas look dark
        view.environment.sceneUnderstanding.options.insert(.occlusion)
        var m = UnlitMaterial(color: .black)
        m.blending = .transparent(opacity: 0.6)
        let plane = ModelEntity(mesh: .generatePlane(width: 20, height: 20), materials: [m])
        plane.position.z = -5
        veil.addChild(plane)
        view.scene.addAnchor(veil)
    }

    private var config: ARWorldTrackingConfiguration {
        let config = ARWorldTrackingConfiguration()
        config.frameSemantics = .sceneDepth
        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) { config.sceneReconstruction = .mesh }
        return config
    }

    func start() {
        let f = DateFormatter()
        f.dateFormat = "yyyyMMdd-HHmmss"
        sessionID = "lidar-" + f.string(from: Date())
        dir = docs.appendingPathComponent(sessionID)
        view.session.run(config, options: .resetSceneReconstruction)
        area = 0; growth = 0
        for sub in ["rgb", "depth", "conf"] {
            try? FileManager.default.createDirectory(at: dir.appendingPathComponent(sub), withIntermediateDirectories: true)
        }
        let meta: [String: Any] = [
            "session": sessionID, "device": UIDevice.current.model, "system": UIDevice.current.systemVersion,
            "fps": Self.fps, "rgb": "jpg, sensor (landscape) orientation", "depth": "float16 little-endian metres, depth_wh",
            "conf": "uint8 0 low, 1 medium, 2 high, depth_wh",
            "pose": "T = ARKit camera-to-world, row-major 4x4; world y up (gravity), metres; camera x right, y up, z back",
            "K": "fx fy cx cy for the saved rgb; K_depth for the depth map"]
        try? JSONSerialization.data(withJSONObject: meta, options: .prettyPrinted).write(to: dir.appendingPathComponent("session.json"))
        lines = handle("frames.jsonl")
        events = handle("events.jsonl")
        frames = 0; bytes = 0; zip = nil; status = ""; t0 = now; last = 0
        recording = true
        UIApplication.shared.isIdleTimerDisabled = true
        mark("room")
    }

    func stop() {
        recording = false
        UIApplication.shared.isIdleTimerDisabled = false
        queue.async { [self] in
            try? lines?.close(); try? events?.close()
            DispatchQueue.main.async { [self] in reviewing = true }
        }
    }

    /// Writes the reviewed per-frame zones (labels.json + frames.jsonl room) if given, then packs the zip.
    func save(_ labels: [String]?) {
        zip = nil
        status = "Packing…"
        queue.async { [self] in
            if let labels { relabel(labels) }
            let out = Result { try pack(dir) }
            DispatchQueue.main.async { [self] in
                zip = try? out.get()
                status = zip.map { "Ready: \(ByteCountFormatter.string(fromByteCount: size($0), countStyle: .file))" }
                    ?? "Packing failed"
            }
        }
    }

    func sessions() -> [String] {
        let all = (try? FileManager.default.contentsOfDirectory(atPath: docs.path)) ?? []
        return all.filter { FileManager.default.fileExists(atPath: docs.appendingPathComponent("\($0)/frames.jsonl").path) }
            .sorted(by: >)
    }

    func open(_ id: String) {
        sessionID = id
        dir = docs.appendingPathComponent(id)
        let z = docs.appendingPathComponent(id + ".zip")
        zip = FileManager.default.fileExists(atPath: z.path) ? z : nil
        status = "Opened \(id)"
        if zip == nil { save(nil) }
    }

    func delete(_ id: String) {
        try? FileManager.default.removeItem(at: docs.appendingPathComponent(id))
        try? FileManager.default.removeItem(at: docs.appendingPathComponent(id + ".zip"))
        if id == sessionID { sessionID = ""; dir = nil; zip = nil; status = "" }
    }

    func enter(_ name: String) {
        room = name
        if recording { mark("room") }
    }

    func mark(_ event: String) {
        let line: [String: Any] = ["t": now - t0, "i": frames, "event": event, "room": room, "T": pose]
        queue.async { [self] in write(line, to: events) }
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let state: String
        switch frame.camera.trackingState {
        case .normal: state = "normal"
        case .notAvailable: state = "notAvailable"
        case .limited(let why): state = "limited:\(why)"
        }
        if state != tracking { tracking = state }
        let T = frame.camera.transform
        pose = (0..<4).flatMap { r in (0..<4).map { c in T[c][r] } }
        now = frame.timestamp
        if now - lastMesh >= 1 {
            lastMesh = now
            let meshes = frame.anchors.compactMap { $0 as? ARMeshAnchor }
            meshQueue.async { [self] in
                let a = meshes.reduce(0) { $0 + Self.area($1.geometry) }
                DispatchQueue.main.async { [self] in growth = a - area; area = a }
            }
        }
        guard recording, !busy, frame.timestamp - last >= 1 / Self.fps, let depth = frame.sceneDepth else { return }
        last = frame.timestamp
        busy = true
        let i = frames, name = String(format: "%06d", frames)
        let res = frame.camera.imageResolution, K = frame.camera.intrinsics
        let s = Self.width / res.width, sd = Double(CVPixelBufferGetWidth(depth.depthMap)) / res.width
        let k = { (s: Double) in [Double(K[0][0]) * s, Double(K[1][1]) * s, Double(K[2][0]) * s, Double(K[2][1]) * s] }
        let line: [String: Any] = [
            "i": i, "t": frame.timestamp - t0, "rgb": "rgb/\(name).jpg", "depth": "depth/\(name).bin", "conf": "conf/\(name).bin",
            "rgb_wh": [Int(Self.width), Int((res.height * s).rounded())], "K": k(s),
            "depth_wh": [CVPixelBufferGetWidth(depth.depthMap), CVPixelBufferGetHeight(depth.depthMap)], "K_depth": k(sd),
            "T": pose, "tracking": state, "room": room]
        let image = CIImage(cvPixelBuffer: frame.capturedImage)
        let depthMap = depth.depthMap, conf = depth.confidenceMap
        queue.async { [self] in
            let scale = CIFilter.lanczosScaleTransform()
            scale.inputImage = image
            scale.scale = Float(s)
            let jpg = ci.jpegRepresentation(of: scale.outputImage!, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!,
                                            options: [CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String): 0.6]) ?? Data()
            let d = plane(depthMap, Float32.self).map { Float16($0) }.withUnsafeBytes { Data($0) }
            let c = conf.map { plane($0, UInt8.self) }.map { Data($0) } ?? Data()
            try? jpg.write(to: dir.appendingPathComponent("rgb/\(name).jpg"))
            try? d.write(to: dir.appendingPathComponent("depth/\(name).bin"))
            try? c.write(to: dir.appendingPathComponent("conf/\(name).bin"))
            write(line, to: lines)
            DispatchQueue.main.async { [self] in
                frames += 1
                bytes += jpg.count + d.count + c.count
                busy = false
            }
        }
    }

    func upload(key: String) {
        guard let zip else { return }
        var url = URLComponents(string: "https://game.adtomic.co/upload")!
        url.queryItems = [.init(name: "session", value: sessionID), .init(name: "room", value: "lidar"), .init(name: "ext", value: "zip")]
        var req = URLRequest(url: url.url!, timeoutInterval: 900)
        req.httpMethod = "POST"
        req.setValue(key, forHTTPHeaderField: "X-API-Key")
        req.setValue("application/zip", forHTTPHeaderField: "Content-Type")
        let task = URLSession.shared.uploadTask(with: req, fromFile: zip) { [self] _, resp, err in
            let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
            DispatchQueue.main.async { [self] in
                progress = nil
                status = err?.localizedDescription ?? (code == 200 ? "Uploaded \(sessionID)" : "Upload failed: HTTP \(code)")
            }
        }
        progress = task.progress.observe(\.fractionCompleted) { [self] p, _ in
            DispatchQueue.main.async { [self] in status = "Uploading \(Int(p.fractionCompleted * 100)) %" }
        }
        task.resume()
    }

    private static func area(_ g: ARMeshGeometry) -> Double {
        let v = g.vertices, base = v.buffer.contents() + v.offset
        let idx = g.faces.buffer.contents().assumingMemoryBound(to: UInt32.self)
        func p(_ j: UInt32) -> SIMD3<Float> {
            let q = (base + Int(j) * v.stride).assumingMemoryBound(to: Float.self)
            return SIMD3(q[0], q[1], q[2])
        }
        var sum: Float = 0
        for f in 0..<g.faces.count {
            let a = p(idx[3 * f])
            sum += simd_length(simd_cross(p(idx[3 * f + 1]) - a, p(idx[3 * f + 2]) - a)) / 2
        }
        return Double(sum)
    }

    private func relabel(_ labels: [String]) {
        let url = dir.appendingPathComponent("frames.jsonl")
        var out = Data(), ranges: [[String: Any]] = []
        let lines = ((try? String(contentsOf: url)) ?? "").split(separator: "\n")
        for (i, s) in lines.enumerated() where i < labels.count {
            guard let obj = (try? JSONSerialization.jsonObject(with: Data(s.utf8))) as? [String: Any],
                  let data = try? JSONSerialization.data(withJSONObject: obj.merging(["room": labels[i]]) { $1 }) else { continue }
            out.append(data + [0x0A])
            let n = obj["i"] as? Int ?? i
            if ranges.last?["room"] as? String == labels[i] { ranges[ranges.count - 1]["to"] = n }
            else { ranges.append(["from": n, "to": n, "room": labels[i]]) }
        }
        try? out.write(to: url)
        let doc: [String: Any] = ["ranges": ranges, "note": "zones fixed after recording; frame i ranges (inclusive) override room"]
        try? JSONSerialization.data(withJSONObject: doc, options: .prettyPrinted).write(to: dir.appendingPathComponent("labels.json"))
    }

    private func handle(_ file: String) -> FileHandle? {
        let url = dir.appendingPathComponent(file)
        FileManager.default.createFile(atPath: url.path, contents: nil)
        return try? FileHandle(forWritingTo: url)
    }

    private func write(_ obj: [String: Any], to h: FileHandle?) {
        guard var data = try? JSONSerialization.data(withJSONObject: obj) else { return }
        data.append(0x0A)
        h?.write(data)
    }

    private func plane<T>(_ buf: CVPixelBuffer, _: T.Type) -> [T] {
        CVPixelBufferLockBaseAddress(buf, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buf, .readOnly) }
        let w = CVPixelBufferGetWidth(buf), h = CVPixelBufferGetHeight(buf), row = CVPixelBufferGetBytesPerRow(buf)
        let base = CVPixelBufferGetBaseAddress(buf)!
        var out = [T]()
        out.reserveCapacity(w * h)
        for y in 0..<h {
            out.append(contentsOf: UnsafeBufferPointer(start: (base + y * row).assumingMemoryBound(to: T.self), count: w))
        }
        return out
    }

    private func pack(_ dir: URL) throws -> URL {
        let dst = dir.deletingLastPathComponent().appendingPathComponent(dir.lastPathComponent + ".zip")
        var err: NSError?, moveErr: Error?
        NSFileCoordinator().coordinate(readingItemAt: dir, options: .forUploading, error: &err) { tmp in
            try? FileManager.default.removeItem(at: dst)
            do { try FileManager.default.moveItem(at: tmp, to: dst) } catch { moveErr = error }
        }
        if let e = err ?? moveErr { throw e }
        return dst
    }

    private func size(_ url: URL) -> Int64 {
        Int64((try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
    }
}
