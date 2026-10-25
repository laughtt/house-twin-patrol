import ARKit
import CoreImage
import CoreImage.CIFilterBuiltins
import RealityKit

struct Tag: Identifiable {
    let i: Int
    var label: String
    let uv: [Double]
    let room: String
    var id: Int { i }
    var dict: [String: Any] { ["i": i, "label": label, "uv": uv, "room": room] }
    init(i: Int, label: String, uv: [Double], room: String) { self.i = i; self.label = label; self.uv = uv; self.room = room }
    init?(_ d: [String: Any]) {
        guard let i = d["i"] as? Int, let label = d["label"] as? String else { return nil }
        self.init(i: i, label: label, uv: d["uv"] as? [Double] ?? [0.5, 0.5], room: d["room"] as? String ?? "")
    }
}

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
    @Published var height: Double?
    @Published var hold = 0.0
    @Published var walls: [String: Double] = [:]
    @Published var sectors: [String: Set<Int>] = [:]
    @Published var facing: Int?
    @Published var lowFast = false
    @Published var lowTilted = false
    @Published var tags: [Tag] = []
    @Published var walked: [String: Double] = [:]
    @Published var sinceZone = 0.0
    @Published var scoring = false
    @Published var uploading = false
    @Published var sent = 0.0
    @Published var uploaded = false
    private(set) var duration = 0.0
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
    private var ref: (t: TimeInterval, p: SIMD3<Float>, q: simd_quatf)?
    private var floorY: Float?
    private var floorSure = false
    private var stillSince: TimeInterval?, lowSince: TimeInterval = 0, lowSector: Int?
    private var forceNext = false, tagShot: [String: Any]?, tagLabel: String?
    private var walkRef: SIMD2<Float>?

    func run() {
        guard supported else { return }
        view.session.run(config)
        guard view.session.delegate == nil else { return }  // back from another tab: the veil is already set up
        view.session.delegate = self
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

    func pause() {
        if supported && !recording { view.session.pause() }
    }

    private var config: ARWorldTrackingConfiguration {
        let config = ARWorldTrackingConfiguration()
        config.frameSemantics = .sceneDepth
        config.planeDetection = .horizontal
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
            "K": "fx fy cx cy for the saved rgb; K_depth for the depth map", "format": 2,
            "h": "camera height above the floor (lowest ARKit floor plane), metres, null until a floor is found",
            "events": "room/doorway taps; tag = {t, i, label, room, T, uv}: i = the frame saved for it, uv = crosshair in that rgb (0..1, top-left origin)",
            "labels": "labels.json: ranges = zones fixed after recording; tags = [{i, label, uv, room}] reviewed after recording, they win over the tag events"]
        try? JSONSerialization.data(withJSONObject: meta, options: .prettyPrinted).write(to: dir.appendingPathComponent("session.json"))
        lines = handle("frames.jsonl")
        events = handle("events.jsonl")
        frames = 0; bytes = 0; zip = nil; status = ""; t0 = now; last = 0
        walls = [:]; sectors = [:]; tags = []; forceNext = false; tagShot = nil
        walked = [:]; sinceZone = 0; walkRef = nil; uploaded = false
        recording = true
        UIApplication.shared.isIdleTimerDisabled = true
        mark("room")
    }

    func stop() {
        recording = false
        UIApplication.shared.isIdleTimerDisabled = false
        duration = now - t0
        let zones = walked.keys.sorted(), r1 = { (x: Double) in (x * 10).rounded() / 10 }
        let score: [String: Any] = [
            "zones": zones.count, "walked_m": walked.mapValues(r1), "duration_s": r1(duration), "frames": frames, "mb": bytes / 1_000_000,
            "sectors": Dictionary(uniqueKeysWithValues: zones.map { ($0, sectors[$0]?.count ?? 0) }),
            "tags": Dictionary(uniqueKeysWithValues: zones.map { z in (z, tags.filter { $0.room == z }.count) })]
        queue.async { [self] in
            try? lines?.close(); try? events?.close()
            let url = dir.appendingPathComponent("session.json")
            var meta = (try? JSONSerialization.jsonObject(with: Data(contentsOf: url))) as? [String: Any] ?? [:]
            meta["score"] = score
            try? JSONSerialization.data(withJSONObject: meta, options: .prettyPrinted).write(to: url)
            DispatchQueue.main.async { [self] in scoring = true }
        }
    }

    /// Writes the reviewed per-frame zones (frames.jsonl room) and/or tags into labels.json if given, then packs the zip.
    func save(_ labels: [String]?, _ tags: [Tag]?) {
        zip = nil
        status = "Packing…"
        queue.async { [self] in
            if labels != nil || tags != nil { relabel(labels, tags ?? []) }
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
        uploaded = false
        sessionID = id
        dir = docs.appendingPathComponent(id)
        let z = docs.appendingPathComponent(id + ".zip")
        zip = FileManager.default.fileExists(atPath: z.path) ? z : nil
        status = "Opened \(sessionTitle(id))"
        if zip == nil { save(nil, nil) }
    }

    func delete(_ id: String) {
        try? FileManager.default.removeItem(at: docs.appendingPathComponent(id))
        try? FileManager.default.removeItem(at: docs.appendingPathComponent(id + ".zip"))
        if id == sessionID { sessionID = ""; dir = nil; zip = nil; status = ""; uploaded = false }
    }

    func enter(_ name: String) {
        room = name
        if recording { mark("room"); sinceZone = 0 }
    }

    func mark(_ event: String) {
        let line: [String: Any] = ["t": now - t0, "i": frames, "event": event, "room": room, "T": pose]
        queue.async { [self] in write(line, to: events) }
    }

    /// Tag: the next camera frame is saved right away; the event is written once both that frame and the label are known.
    func tag() { forceNext = true; tagShot = nil; tagLabel = nil }

    func label(_ l: String?) {
        guard let l else { forceNext = false; tagShot = nil; tagLabel = nil; return }
        tagLabel = l
        flushTag()
    }

    private func flushTag() {
        guard let shot = tagShot, let l = tagLabel else { return }
        let line = shot.merging(["label": l]) { $1 }
        queue.async { [self] in write(line, to: events) }
        tags.append(Tag(i: shot["i"] as! Int, label: l, uv: shot["uv"] as! [Double], room: shot["room"] as! String))
        status = "Tagged \(l)"
        tagShot = nil; tagLabel = nil
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
        guide(frame)
        if now - lastMesh >= 1 {
            lastMesh = now
            let meshes = frame.anchors.compactMap { $0 as? ARMeshAnchor }
            meshQueue.async { [self] in
                let a = meshes.reduce(0) { $0 + Self.area($1.geometry) }
                DispatchQueue.main.async { [self] in growth = a - area; area = a }
            }
        }
        guard recording, !busy, forceNext || frame.timestamp - last >= 1 / Self.fps, let depth = frame.sceneDepth else { return }
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
            "T": pose, "tracking": state, "room": room,
            "h": floorY.map { (Double(T.columns.3.y - $0) * 100).rounded() / 100 as Any } ?? NSNull()]
        if forceNext {
            forceNext = false
            let c = CGPoint(x: 0.5, y: 0.5).applying(frame.displayTransform(for: .portrait, viewportSize: view.bounds.size).inverted())
            let uv = [c.x, c.y].map { (Double($0) * 1e4).rounded() / 1e4 }
            tagShot = ["t": frame.timestamp - t0, "i": i, "event": "tag", "room": room, "T": pose, "uv": uv]
            flushTag()
        }
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

    func upload() {
        guard let zip, !uploading else { return }
        var req = Auth.shared.request("/upload", ["session": sessionID, "room": "lidar", "ext": "zip"], method: "POST")
        req.timeoutInterval = 900
        req.setValue("application/zip", forHTTPHeaderField: "Content-Type")
        uploading = true
        uploaded = false
        sent = 0
        status = "Uploading…"
        UIApplication.shared.isIdleTimerDisabled = true  // a locked phone kills the upload
        let task = URLSession.shared.uploadTask(with: req, fromFile: zip) { [self] _, resp, err in
            let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
            DispatchQueue.main.async { [self] in
                progress = nil
                uploading = false
                UIApplication.shared.isIdleTimerDisabled = false
                if err == nil && code == 200 { uploaded = true; status = "Uploaded: your map is building" }
                else if code == 401 { status = "Signed out: sign in and tap Upload again"; Auth.shared.signOut() }
                else { status = "Upload failed: \(err.map { ApiError.from($0).localizedDescription } ?? "server error \(code)"). Tap Upload to retry." }
            }
        }
        progress = task.progress.observe(\.fractionCompleted) { [self] p, _ in
            DispatchQueue.main.async { [self] in if uploading { sent = p.fractionCompleted; status = "Uploading \(Int(p.fractionCompleted * 100)) %" } }
        }
        task.resume()
    }

    /// Every 0.2 s: floor and camera height, motion over the last 0.2 s, hold-still ring and the per-zone checklist.
    private func guide(_ frame: ARFrame) {
        let T = frame.camera.transform, p = SIMD3(T.columns.3.x, T.columns.3.y, T.columns.3.z), q = simd_quatf(T)
        guard let r = ref, now - r.t >= 0.2 else { if ref == nil { ref = (now, p, q) }; return }
        ref = (now, p, q)
        let dt = now - r.t, a = (r.q.inverse * q).angle
        let speed = Double(simd_length(p - r.p)) / dt, rot = Double(min(a, 2 * .pi - a)) * 180 / .pi / dt
        let planes = frame.anchors.compactMap { $0 as? ARPlaneAnchor }.filter { $0.alignment == .horizontal }
        // the floor only goes down: a sofa seat or table top must never pass for it (heights would read too low)
        if let y = planes.filter({ $0.classification == .floor }).map(\.transform.columns.3.y).min() {
            floorY = min(floorY ?? y, y)
            floorSure = true
        } else if !floorSure, let y = planes.filter({ $0.planeExtent.width * $0.planeExtent.height >= 0.5 }).map(\.transform.columns.3.y).min() {
            floorY = min(floorY ?? y, y)
        }
        let h = floorY.map { Double(p.y - $0) }
        height = h
        let f = SIMD2(-T.columns.2.x, -T.columns.2.z)  // heading on the floor plane (the camera looks along -z)
        let k = simd_length(f) > 0.5 ? Int((Double(atan2(f.y, f.x)) / (.pi / 4) + 8).rounded(.down)) % 8 : nil
        if k != facing { facing = k }
        guard recording else { hold = 0; lowFast = false; lowTilted = false; return }
        if speed < 0.05 && rot < 10 {
            let s = stillSince ?? r.t
            stillSince = s
            let v = min((now - s) / 1.5, 1)
            if v == 1 && hold < 1 { UIImpactFeedbackGenerator(style: .light).impactOccurred() }
            hold = v
        } else { stillSince = nil; hold = 0 }
        if h.map({ $0 >= 0.9 }) ?? true { walls[room, default: 0] += min(dt, 0.5) }
        // distance on the floor in 0.75 m steps, so hand jitter and turning on the spot don't add up; jumps over 5 m are relocalisation
        let xz = SIMD2(p.x, p.z), d = walkRef.map { Double(simd_distance(xz, $0)) } ?? 0
        if walkRef == nil || d >= 0.75 { walkRef = xz }
        let step = d >= 0.75 && d < 5 ? d : 0
        if walked[room] == nil || step > 0 { walked[room, default: 0] += step; sinceZone += step }
        let low = h.map { $0 < 0.6 } ?? false, calm = rot < 30 && speed < 0.15
        let level = abs(asin(Double(-T.columns.2.y))) < 25 * .pi / 180  // the duck looks straight ahead, not at the floor
        if low && !level != lowTilted { lowTilted = low && !level }
        if low && level && !calm != lowFast { lowFast = low && level && !calm }
        if low, level, calm, let k {
            if k != lowSector { lowSector = k; lowSince = r.t }
            else if now - lowSince >= 1, sectors[room]?.contains(k) != true { sectors[room, default: []].insert(k) }
        } else { lowSector = nil }
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

    private func relabel(_ labels: [String]?, _ tags: [Tag]) {
        let url = dir.appendingPathComponent("frames.jsonl")
        var out = Data(), ranges: [[String: Any]] = []
        let lines = ((try? String(contentsOf: url)) ?? "").split(separator: "\n")
        for (i, s) in lines.enumerated() where i < labels?.count ?? lines.count {
            guard let obj = (try? JSONSerialization.jsonObject(with: Data(s.utf8))) as? [String: Any] else { continue }
            let room = labels?[i] ?? obj["room"] as? String ?? ""
            guard let data = try? JSONSerialization.data(withJSONObject: obj.merging(["room": room]) { $1 }) else { continue }
            out.append(data + [0x0A])
            let n = obj["i"] as? Int ?? i
            if ranges.last?["room"] as? String == room { ranges[ranges.count - 1]["to"] = n }
            else { ranges.append(["from": n, "to": n, "room": room]) }
        }
        if labels != nil { try? out.write(to: url) }
        let doc: [String: Any] = ["ranges": ranges, "tags": tags.map(\.dict),
                                  "note": "zones fixed after recording; frame i ranges (inclusive) override room. tags: objects labelled after recording, uv = point in frame i's rgb (0..1, top-left origin); they replace the tag events"]
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
