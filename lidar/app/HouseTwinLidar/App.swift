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
    @AppStorage("tagLabels") var tagLabels = "Fridge,Oven,Stove,Sink,Sofa,Bed,TV,Table,Toilet,Washer,Wardrobe,Door,Window,Plant,Other"
    @State var settings = false
    @State var sessions = false
    @State var tagging = false

    var body: some View {
        if !rec.supported {
            Text("This device has no LiDAR depth (ARKit sceneDepth is not supported).\nUse an iPhone 12 Pro or newer Pro model, or an iPad Pro.")
                .multilineTextAlignment(.center).padding()
        } else {
            ZStack {
                Preview(view: rec.view).ignoresSafeArea()
                if rec.recording {
                    Color.clear.overlay {
                        ZStack {
                            Circle().stroke(.white.opacity(0.3), lineWidth: 4)
                            Circle().trim(from: 0, to: rec.hold).stroke(rec.hold >= 1 ? Color.green : Color.white, style: StrokeStyle(lineWidth: 4, lineCap: .round))
                                .rotationEffect(.degrees(-90)).animation(.linear(duration: 0.2), value: rec.hold)
                            Image(systemName: "plus").font(.title2.weight(.light))
                        }
                        .foregroundColor(.white).frame(width: 54, height: 54).shadow(radius: 2)
                    }
                    .ignoresSafeArea().allowsHitTesting(false)
                }
                VStack(spacing: 10) {
                    HStack(alignment: .top) {
                        VStack(alignment: .leading) {
                            Text("\(rec.frames) frames · \(rec.bytes / 1_000_000) MB")
                            Text("Room: \(rec.room)")
                            Text("Tracking: \(rec.tracking)")
                            Text(rec.height.map { String(format: "Height %.2f m", $0) } ?? "Height: finding the floor…")
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
                    if rec.recording { checklist }
                    if let nudge {
                        Text(nudge).font(.callout.bold()).padding(8).frame(maxWidth: .infinity)
                            .background(.cyan).foregroundColor(.black).cornerRadius(8)
                    } else if rec.recording && veil {
                        Text(rec.growth > 0.3 ? "New surfaces lighting up: keep sweeping slowly"
                                              : "Nothing new here: aim at the dark parts or move on")
                            .font(.callout.bold()).padding(6).background(.black.opacity(0.5)).foregroundColor(.white).cornerRadius(8)
                    }
                    if rec.bytes > 500_000_000 {
                        Text("Over 500 MB: stop soon (upload limit 600 MB)").font(.callout.bold())
                            .padding(8).background(.red).foregroundColor(.white).cornerRadius(8)
                    }
                    Spacer()
                    LazyVGrid(columns: [GridItem(), GridItem(), GridItem()]) {
                        ForEach(Self.items(zones), id: \.self) { z in
                            Button(z) { rec.enter(z) }
                                .frame(maxWidth: .infinity, minHeight: 44)
                                .background(z == rec.room ? Color.blue : Color.black.opacity(0.55))
                                .foregroundColor(.white).cornerRadius(8)
                        }
                    }
                    if rec.recording {
                        HStack {
                            Button("Doorway") { rec.mark("doorway") }
                                .frame(maxWidth: .infinity, minHeight: 40).background(.yellow).foregroundColor(.black).cornerRadius(8)
                            Button { rec.tag(); tagging = true } label: { Label("Tag", systemImage: "scope") }
                                .frame(maxWidth: .infinity, minHeight: 40).background(.mint).foregroundColor(.black).cornerRadius(8)
                        }
                    }
                    Button(rec.recording ? "Stop" : "Start") { rec.recording ? rec.stop() : rec.start() }
                        .font(.largeTitle.bold()).frame(maxWidth: .infinity, minHeight: 80)
                        .background(rec.recording ? Color.red : Color.green).foregroundColor(.white).cornerRadius(16)
                    if let zip = rec.zip, !rec.recording {
                        HStack {
                            Button { rec.reviewing = true } label: { Label("Zones & tags", systemImage: "slider.horizontal.below.rectangle") }
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
            .sheet(isPresented: $settings) { Settings(zones: $zones, key: $key, veil: $veil, tagLabels: $tagLabels) }
            .sheet(isPresented: $sessions) { Sessions(rec: rec) }
            .sheet(isPresented: $tagging) { LabelGrid(labels: Self.items(tagLabels)) { rec.label($0); tagging = false } }
            .fullScreenCover(isPresented: $rec.reviewing) {
                Review(dir: rec.dir, zones: Self.items(zones), labels: Self.items(tagLabels)) { rec.reviewing = false; rec.save($0, $1) }
            }
        }
    }

    var nudge: String? {
        guard rec.recording else { return nil }
        if rec.lowFast { return "Slower — hold still a second in each direction" }
        if rec.walls[rec.room, default: 0] >= 25 && rec.sectors[rec.room, default: []].count < 4 { return "Now go low ⬇︎ duck height" }
        return nil
    }

    var checklist: some View {
        let walls = rec.walls[rec.room, default: 0], low = rec.sectors[rec.room, default: []]
        let tags = rec.tags.filter { $0.room == rec.room }.count
        let tip = walls < 25 ? "Sweep every wall at chest height"
            : low.count < 8 ? "Go down to duck height: phone below your knee (under 60 cm), turn slowly all the way round"
            : tags < 3 ? "Aim the crosshair at big things that never move (fridge, oven, sofa, bed, toilet, TV) and tap Tag"
            : "\(rec.room) done: next zone"
        return VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 14) {
                Label("Walls \(min(Int(walls), 25))/25 s", systemImage: walls >= 25 ? "checkmark.circle.fill" : "circle")
                HStack(spacing: 5) {
                    ZStack {
                        ForEach(0..<8) { k in
                            Circle().trim(from: CGFloat(k) / 8 + 0.012, to: CGFloat(k + 1) / 8 - 0.012)
                                .stroke(low.contains(k) ? Color.green : k == rec.facing ? Color.yellow : Color.white.opacity(0.35), lineWidth: 4)
                        }
                    }
                    .frame(width: 20, height: 20)
                    Text("Low \(low.count)/8")
                }
                Label("Tags \(tags)/3", systemImage: tags >= 3 ? "checkmark.circle.fill" : "circle")
            }
            .font(.subheadline.bold().monospacedDigit())
            Text(tip).font(.caption)
        }
        .padding(8).frame(maxWidth: .infinity, alignment: .leading)
        .background(.black.opacity(0.5)).foregroundColor(.white).cornerRadius(10)
    }

    static func items(_ list: String) -> [String] {
        list.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
    }
}

struct LabelGrid: View {
    let labels: [String]
    let pick: (String?) -> Void

    var body: some View {
        VStack(spacing: 12) {
            Text("What is it?").font(.headline)
            LazyVGrid(columns: [GridItem(), GridItem(), GridItem()]) {
                ForEach(labels, id: \.self) { l in
                    Button(l) { pick(l) }
                        .frame(maxWidth: .infinity, minHeight: 44).background(.blue).foregroundColor(.white).cornerRadius(8)
                }
            }
            Button("Cancel") { pick(nil) }.frame(maxWidth: .infinity, minHeight: 44)
        }
        .padding().presentationDetents([.medium]).interactiveDismissDisabled()
    }
}

struct Settings: View {
    @Binding var zones: String
    @Binding var key: String
    @Binding var veil: Bool
    @Binding var tagLabels: String
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
                Section(header: Text("Tag labels (comma-separated)")) {
                    TextField("Fridge,Oven,…", text: $tagLabels)
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
/// change (or the selected range). Then, if there are tags, check their labels.
struct Review: View {
    let dir: URL
    let zones: [String]
    let labels: [String]
    let done: ([String]?, [Tag]?) -> Void
    @State var tags: [Tag] = []
    @State var crops: [Int: UIImage] = [:]
    @State var tagStep = false
    @State var keep: [String]?
    @State var editing: Tag?
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
                Button("Skip") { if tags.isEmpty { done(nil, nil) } else { keep = nil; tagStep = true } }
                    .frame(maxWidth: .infinity, minHeight: 48).background(.gray.opacity(0.4)).foregroundColor(.white).cornerRadius(10)
                Button(tags.isEmpty ? "Save" : "Next") { if tags.isEmpty { done(rooms, []) } else { keep = rooms; tagStep = true } }.font(.headline)
                    .frame(maxWidth: .infinity, minHeight: 48).background(.green).foregroundColor(.white).cornerRadius(10)
            }
        }
        .padding()
        .background(Color.black.ignoresSafeArea())
        .overlay { if tagStep { tagList } }
        .sheet(item: $editing) { t in
            LabelGrid(labels: labels) { l in
                if let l, let k = tags.firstIndex(where: { $0.id == t.id }) { tags[k].label = l }
                editing = nil
            }
        }
        .task { await load() }
    }

    var tagList: some View {
        VStack(spacing: 10) {
            Text("Tags: tap to change the label, swipe to delete").font(.headline).foregroundColor(.white)
            List {
                ForEach(tags) { t in
                    HStack(spacing: 12) {
                        if let img = crops[t.i] { Image(uiImage: img).resizable().scaledToFill().frame(width: 64, height: 64).clipped().cornerRadius(6) }
                        else { Color.gray.frame(width: 64, height: 64).cornerRadius(6) }
                        VStack(alignment: .leading) {
                            Text(t.label).font(.title3.bold())
                            Text("\(t.room) · frame \(t.i)").font(.caption).foregroundColor(.secondary)
                        }
                        Spacer()
                    }
                    .contentShape(Rectangle())
                    .onTapGesture { editing = t }
                }
                .onDelete { tags.remove(atOffsets: $0) }
            }
            .listStyle(.plain).environment(\.colorScheme, .dark)
            .overlay { if tags.isEmpty { Text("No tags left").foregroundColor(.secondary) } }
            HStack {
                Button("Back") { tagStep = false }
                    .frame(maxWidth: .infinity, minHeight: 48).background(.gray.opacity(0.4)).foregroundColor(.white).cornerRadius(10)
                Button("Save") { done(keep, tags) }.font(.headline)
                    .frame(maxWidth: .infinity, minHeight: 48).background(.green).foregroundColor(.white).cornerRadius(10)
            }
        }
        .padding()
        .background(Color.black.ignoresSafeArea())
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
        let json = { (file: String) in
            ((try? String(contentsOf: dir.appendingPathComponent(file))) ?? "").split(separator: "\n")
                .compactMap { (try? JSONSerialization.jsonObject(with: Data($0.utf8))) as? [String: Any] }
        }
        let saved = (try? JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("labels.json")))) as? [String: Any]
        tags = ((saved?["tags"] as? [[String: Any]]) ?? json("events.jsonl").filter { $0["event"] as? String == "tag" }).compactMap { Tag($0) }
        for t in tags {
            let path = dir.appendingPathComponent(String(format: "rgb/%06d.jpg", t.i)).path, uv = t.uv
            let cg = await Task.detached { () -> CGImage? in
                guard let img = UIImage(contentsOfFile: path)?.cgImage, uv.count == 2 else { return nil }
                let w = Double(img.width), h = Double(img.height), s = h / 2
                return img.cropping(to: CGRect(x: min(max(uv[0] * w - s / 2, 0), w - s), y: min(max(uv[1] * h - s / 2, 0), h - s), width: s, height: s))
            }.value
            crops[t.i] = cg.map { UIImage(cgImage: $0, scale: 1, orientation: .right) }
        }
        let lines = json("frames.jsonl")
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
