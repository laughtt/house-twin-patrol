import RealityKit
import SwiftUI

@main
struct HouseTwinLidarApp: App {
    var body: some SwiftUI.Scene { WindowGroup { ContentView() } }
}

struct Preview: UIViewRepresentable {
    let view: ARView
    func makeUIView(context: Context) -> ARView { view }
    func updateUIView(_ uiView: ARView, context: Context) {}
}

struct ContentView: View {
    @StateObject var rec = Recorder()
    @AppStorage("zones") var zones = "Kitchen,Living room,Dining,Bedroom,Bathroom,Hallway"
    @AppStorage("apiKey") var key = ""
    @AppStorage("veil") var veil = true
    @State var settings = false
    @State var sessions = false

    var body: some View {
        if !rec.supported {
            Text("This device has no LiDAR depth (ARKit sceneDepth is not supported).\nUse an iPhone 12 Pro or newer Pro model, or an iPad Pro.")
                .multilineTextAlignment(.center).padding()
        } else {
            ZStack {
                Preview(view: rec.view).ignoresSafeArea()
                VStack(spacing: 10) {
                    HStack(alignment: .top) {
                        VStack(alignment: .leading) {
                            Text("\(rec.frames) frames · \(rec.bytes / 1_000_000) MB")
                            Text("Room: \(rec.room)")
                            Text("Tracking: \(rec.tracking)")
                            if veil { Text("Scanned: \(Int(rec.area)) m²") }
                        }
                        .font(.headline.monospacedDigit())
                        Spacer()
                        Button { sessions = true } label: { Image(systemName: "folder.fill").font(.title2) }
                            .disabled(rec.recording).padding(.trailing, 8)
                        Button { settings = true } label: { Image(systemName: "gearshape.fill").font(.title2) }
                            .disabled(rec.recording)
                    }
                    .padding(10).background(.black.opacity(0.5)).foregroundColor(.white).cornerRadius(10)
                    if rec.tracking != "normal" {
                        Text("Tracking limited: slow down, add light, point at textured surfaces")
                            .font(.callout.bold()).padding(8).frame(maxWidth: .infinity)
                            .background(.orange).foregroundColor(.black).cornerRadius(8)
                    }
                    if rec.recording && veil {
                        Text(rec.growth > 0.3 ? "New surfaces lighting up: keep sweeping slowly"
                                              : "Nothing new here: aim at the dark parts or move on")
                            .font(.callout.bold()).padding(6).background(.black.opacity(0.5)).foregroundColor(.white).cornerRadius(8)
                    }
                    if rec.bytes > 250_000_000 {
                        Text("Over 250 MB: stop soon (upload limit 300 MB)").font(.callout.bold())
                            .padding(8).background(.red).foregroundColor(.white).cornerRadius(8)
                    }
                    Spacer()
                    LazyVGrid(columns: [GridItem(), GridItem(), GridItem()]) {
                        ForEach(zoneList, id: \.self) { z in
                            Button(z) { rec.enter(z) }
                                .frame(maxWidth: .infinity, minHeight: 44)
                                .background(z == rec.room ? Color.blue : Color.black.opacity(0.55))
                                .foregroundColor(.white).cornerRadius(8)
                        }
                    }
                    if rec.recording {
                        Button("Doorway") { rec.mark("doorway") }
                            .frame(maxWidth: .infinity, minHeight: 40).background(.yellow).foregroundColor(.black).cornerRadius(8)
                    }
                    Button(rec.recording ? "Stop" : "Start") { rec.recording ? rec.stop() : rec.start() }
                        .font(.largeTitle.bold()).frame(maxWidth: .infinity, minHeight: 80)
                        .background(rec.recording ? Color.red : Color.green).foregroundColor(.white).cornerRadius(16)
                    if let zip = rec.zip, !rec.recording {
                        HStack {
                            Button { rec.reviewing = true } label: { Label("Fix zones", systemImage: "slider.horizontal.below.rectangle") }
                                .frame(maxWidth: .infinity, minHeight: 44).background(.white).cornerRadius(8)
                            ShareLink(item: zip) { Label("Share / AirDrop", systemImage: "square.and.arrow.up") }
                                .frame(maxWidth: .infinity, minHeight: 44).background(.white).cornerRadius(8)
                        }
                        Button { key.isEmpty ? settings = true : rec.upload(key: key) } label: {
                            Label(key.isEmpty ? "Upload — add your link in ⚙️ first" : "Upload", systemImage: "icloud.and.arrow.up")
                        }
                        .frame(maxWidth: .infinity, minHeight: 44).background(.white).cornerRadius(8)
                    }
                    if !rec.status.isEmpty {
                        Text(rec.status).font(.callout).padding(6).background(.black.opacity(0.6)).foregroundColor(.white).cornerRadius(6)
                    }
                }
                .padding()
            }
            .onAppear { rec.run(); rec.veil.isEnabled = veil }
            .onChange(of: veil) { rec.veil.isEnabled = $0 }
            .sheet(isPresented: $settings) { Settings(zones: $zones, key: $key, veil: $veil) }
            .sheet(isPresented: $sessions) { Sessions(rec: rec) }
            .fullScreenCover(isPresented: $rec.reviewing) {
                Review(dir: rec.dir, zones: zoneList) { rec.reviewing = false; rec.save($0) }
            }
        }
    }

    var zoneList: [String] {
        zones.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
    }
}

struct Settings: View {
    @Binding var zones: String
    @Binding var key: String
    @Binding var veil: Bool
    @State var link = ""
    @Environment(\.dismiss) var dismiss

    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Phone link (https://game.adtomic.co/?k=…)"),
                        footer: Text(key.isEmpty ? "No key yet: Upload stays off." : "Key saved (…\(key.suffix(4))).")) {
                    TextField("Paste your phone link", text: $link)
                        .textInputAutocapitalization(.never).autocorrectionDisabled()
                        .onChange(of: link) { v in
                            let s = v.trimmingCharacters(in: .whitespacesAndNewlines)
                            if let k = URLComponents(string: s)?.queryItems?.first(where: { $0.name == "k" })?.value, !k.isEmpty {
                                key = k
                            }
                        }
                }
                Section(header: Text("Zones (comma-separated)")) {
                    TextField("Kitchen,Living room,…", text: $zones)
                }
                Section(footer: Text("Dark = not scanned yet; surfaces light up once the LiDAR mesh covers them.")) {
                    Toggle("Coverage veil", isOn: $veil)
                }
            }
            .navigationTitle("Settings")
            .toolbar { Button("Done") { dismiss() } }
        }
    }
}


struct Sessions: View {
    @ObservedObject var rec: Recorder
    @State var ids: [String] = []
    @Environment(\.dismiss) var dismiss

    var body: some View {
        NavigationView {
            List {
                ForEach(ids, id: \.self) { id in Button(id) { rec.open(id); dismiss() } }
                    .onDelete { $0.forEach { rec.delete(ids[$0]) }; ids = rec.sessions() }
            }
            .overlay { if ids.isEmpty { Text("No recordings yet").foregroundColor(.secondary) } }
            .navigationTitle("Sessions")
            .toolbar { Button("Done") { dismiss() } }
            .onAppear { ids = rec.sessions() }
        }
    }
}

/// Fix the zones after recording: scrub the strip or the thumbnails, tap a zone to relabel from here to the next
/// change (or the selected range).
struct Review: View {
    let dir: URL
    let zones: [String]
    let done: ([String]?) -> Void
    @State var rooms: [String] = []
    @State var orig: [String] = []
    @State var thumbs: [UIImage?] = []
    @State var cursor = 0
    @State var anchor: Int?
    static let palette: [Color] = [.blue, .orange, .green, .purple, .pink, .teal, .yellow, .red, .brown, .mint, .indigo, .cyan]

    var body: some View {
        VStack(spacing: 10) {
            ZStack(alignment: .bottomLeading) {
                Color.black
                if let img = thumbs.indices.contains(cursor / 4) ? thumbs[cursor / 4] : nil {
                    Image(uiImage: img).resizable().scaledToFit().frame(maxWidth: .infinity, maxHeight: .infinity)
                }
                if !rooms.isEmpty {
                    Text(String(format: "%d:%02d  ", cursor / 240, cursor / 4 % 60) + rooms[cursor])
                        .font(.title3.bold().monospacedDigit()).padding(8).background(color(rooms[cursor])).foregroundColor(.white)
                }
            }
            .cornerRadius(10)
            GeometryReader { g in
                Canvas { ctx, size in
                    let w = size.width / CGFloat(max(rooms.count, 1))
                    for (i, r) in rooms.enumerated() {
                        ctx.fill(Path(CGRect(x: CGFloat(i) * w, y: 0, width: w + 0.5, height: size.height)), with: .color(color(r)))
                    }
                    if let a = anchor {
                        let x0 = CGFloat(min(a, cursor)) * w, x1 = CGFloat(max(a, cursor) + 1) * w
                        ctx.fill(Path(CGRect(x: x0, y: 0, width: x1 - x0, height: size.height)), with: .color(.white.opacity(0.45)))
                    }
                    ctx.fill(Path(CGRect(x: (CGFloat(cursor) + 0.5) * w - 1.5, y: 0, width: 3, height: size.height)), with: .color(.white))
                }
                .contentShape(Rectangle())
                .gesture(DragGesture(minimumDistance: 0).onChanged {
                    cursor = min(max(Int($0.location.x / g.size.width * CGFloat(rooms.count)), 0), max(rooms.count - 1, 0))
                })
            }
            .frame(height: 44).cornerRadius(6)
            ScrollViewReader { sv in
                ScrollView(.horizontal, showsIndicators: false) {
                    LazyHStack(spacing: 3) {
                        ForEach(thumbs.indices, id: \.self) { k in
                            VStack(spacing: 0) {
                                if let img = thumbs[k] { Image(uiImage: img).resizable().scaledToFill() } else { Color.gray }
                                color(rooms[k * 4]).frame(height: 8)
                            }
                            .frame(width: 45, height: 68).clipped()
                            .overlay(RoundedRectangle(cornerRadius: 3).stroke(.white, lineWidth: cursor / 4 == k ? 3 : 0))
                            .id(k)
                            .onTapGesture { cursor = k * 4 }
                        }
                    }
                }
                .frame(height: 68)
                .onChange(of: cursor) { c in sv.scrollTo(c / 4, anchor: .center) }
            }
            HStack {
                Button { cursor = max(cursor - 1, 0) } label: { Image(systemName: "chevron.left").frame(width: 44, height: 40) }
                Button(anchor == nil ? "Select range from here" : "Cancel range") { anchor = anchor == nil ? cursor : nil }
                    .frame(maxWidth: .infinity, minHeight: 40).background(anchor == nil ? Color.gray.opacity(0.3) : Color.white).cornerRadius(8)
                Button { cursor = min(cursor + 1, max(rooms.count - 1, 0)) } label: { Image(systemName: "chevron.right").frame(width: 44, height: 40) }
            }
            Text(anchor == nil ? "Tap a zone: it applies from here to the next change" : "Tap a zone: it applies to the selected range")
                .font(.caption).foregroundColor(.white.opacity(0.8))
            LazyVGrid(columns: [GridItem(), GridItem(), GridItem()]) {
                ForEach(zones, id: \.self) { z in
                    Button(z) { assign(z) }
                        .frame(maxWidth: .infinity, minHeight: 40).background(color(z)).foregroundColor(.white).cornerRadius(8)
                }
            }
            HStack {
                Button("Undo all") { rooms = orig; anchor = nil }
                    .frame(maxWidth: .infinity, minHeight: 48).background(.gray.opacity(0.4)).foregroundColor(.white).cornerRadius(10)
                Button("Skip") { done(nil) }
                    .frame(maxWidth: .infinity, minHeight: 48).background(.gray.opacity(0.4)).foregroundColor(.white).cornerRadius(10)
                Button("Save") { done(rooms) }.font(.headline)
                    .frame(maxWidth: .infinity, minHeight: 48).background(.green).foregroundColor(.white).cornerRadius(10)
            }
        }
        .padding()
        .background(Color.black.ignoresSafeArea())
        .task { await load() }
    }

    func color(_ zone: String) -> Color {
        let i = zones.firstIndex(of: zone) ?? zones.count + zone.unicodeScalars.reduce(0) { $0 + Int($1.value) }
        return Self.palette[i % Self.palette.count]
    }

    func assign(_ zone: String) {
        guard !rooms.isEmpty else { return }
        let lo = min(anchor ?? cursor, cursor)
        let hi = anchor.map { max($0, cursor) }
            ?? ((cursor..<rooms.count).first { rooms[$0] != rooms[cursor] } ?? rooms.count) - 1
        for i in lo...hi { rooms[i] = zone }
        anchor = nil
    }

    func load() async {
        let text = (try? String(contentsOf: dir.appendingPathComponent("frames.jsonl"))) ?? ""
        let lines = text.split(separator: "\n").compactMap { (try? JSONSerialization.jsonObject(with: Data($0.utf8))) as? [String: Any] }
        rooms = lines.map { $0["room"] as? String ?? "" }
        orig = rooms
        thumbs = Array(repeating: nil, count: (lines.count + 3) / 4)
        for k in thumbs.indices {
            let path = dir.appendingPathComponent(lines[k * 4]["rgb"] as? String ?? "").path
            let cg = await Task.detached { UIImage(contentsOfFile: path)?.preparingThumbnail(of: CGSize(width: 320, height: 240))?.cgImage }.value
            thumbs[k] = cg.map { UIImage(cgImage: $0, scale: 1, orientation: .right) }  // sensor landscape → portrait
        }
    }
}
