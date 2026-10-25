import RealityKit
import SwiftUI

@main
struct HouseTwinLidarApp: App {
    @StateObject var auth = Auth.shared
    @StateObject var rec = Recorder()  // outlives a sign-out, so a recording is never lost

    var body: some SwiftUI.Scene {
        WindowGroup {
            Group { if auth.token == nil { LoginView() } else { Main(rec: rec) } }
                .environmentObject(auth)
        }
    }
}

struct Main: View {
    @ObservedObject var rec: Recorder
    @StateObject var maps = Maps()
    @AppStorage("zones") var zones = "Kitchen,Living room,Dining,Bedroom,Bathroom,Hallway"
    @AppStorage("tagLabels") var tagLabels = "Fridge,Oven,Stove,Sink,Sofa,Bed,TV,Table,Toilet,Washer,Wardrobe,Door,Window,Plant,Other"
    @AppStorage("veil") var veil = true
    @State var tab = 0
    @Environment(\.scenePhase) var phase

    var body: some View {
        TabView(selection: $tab) {
            ContentView(rec: rec, zones: zones, tagLabels: tagLabels, veil: veil) { tab = 1 }
                .tabItem { Label("Record", systemImage: "camera.viewfinder") }.tag(0)
            MapsView(maps: maps, rec: rec)
                .tabItem { Label("My maps", systemImage: "map") }.tag(1)
            Settings(zones: $zones, veil: $veil, tagLabels: $tagLabels)
                .tabItem { Label("Settings", systemImage: "gearshape") }.tag(2)
        }
        .task(id: maps.busy) {
            if !maps.loaded { await maps.refresh() }
            while maps.busy && !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 5_000_000_000)
                await maps.refresh()
            }
        }
        .onChange(of: rec.uploaded) { if $0 { maps.askToNotify(); Task { await maps.refresh() } } }
        .onChange(of: phase) { if $0 == .active { Task { await maps.refresh() } } }
    }
}

struct Fill: ButtonStyle {
    var color: Color = .accentColor
    var text: Color = .white
    var height: CGFloat = 48
    @Environment(\.isEnabled) private var enabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label.font(.headline).multilineTextAlignment(.center)
            .padding(.horizontal, 10).frame(maxWidth: .infinity, minHeight: height)
            .foregroundColor(text).background(color, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .opacity(enabled ? configuration.isPressed ? 0.75 : 1 : 0.45)
    }
}

extension View {
    func card() -> some View {
        padding(12).frame(maxWidth: .infinity, alignment: .leading)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
    }

    func banner(_ color: Color) -> some View {
        font(.callout.weight(.semibold)).padding(10).frame(maxWidth: .infinity, alignment: .leading)
            .background(color, in: RoundedRectangle(cornerRadius: 12, style: .continuous)).foregroundColor(.black)
    }
}

/// An empty or error state: big symbol, title, one line of help, optional retry.
struct Empty: View {
    let icon: String
    let title: String
    let text: String
    var retry: (() -> Void)?

    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: icon).font(.largeTitle).imageScale(.large).foregroundStyle(.secondary).accessibilityHidden(true)
            Text(title).font(.title3.bold())
            Text(text).font(.callout).foregroundStyle(.secondary).multilineTextAlignment(.center)
            if let retry { Button("Try again", action: retry).buttonStyle(.borderedProminent).padding(.top, 4) }
        }
        .padding(32).frame(maxWidth: 420)
    }
}

struct Preview: UIViewRepresentable {
    let view: ARView
    func makeUIView(context: Context) -> ARView { view }
    func updateUIView(_ uiView: ARView, context: Context) {}
}

struct ContentView: View {
    @ObservedObject var rec: Recorder
    let zones: String
    let tagLabels: String
    let veil: Bool
    let showMaps: () -> Void
    @State var sessions = false
    @State var tagging = false

    var body: some View {
        if !rec.supported {
            VStack {
                Empty(icon: "camera.metering.unknown", title: "No LiDAR on this device",
                      text: "Recording needs an iPhone 12 Pro or newer Pro model, or an iPad Pro. You can still open your maps.")
                Button("Open My maps", action: showMaps).buttonStyle(Fill()).frame(maxWidth: 320)
            }
        } else {
            NavigationStack { recorder.toolbar(.hidden, for: .navigationBar) }
                .toolbar(rec.recording ? .hidden : .visible, for: .tabBar)
        }
    }

    var recorder: some View {
        ZStack {
            Preview(view: rec.view).ignoresSafeArea()
            if rec.recording {
                ZStack {
                    Circle().stroke(.white.opacity(0.3), lineWidth: 4)
                    Circle().trim(from: 0, to: rec.hold).stroke(rec.hold >= 1 ? Color.green : Color.white, style: StrokeStyle(lineWidth: 4, lineCap: .round))
                        .rotationEffect(.degrees(-90)).animation(.linear(duration: 0.2), value: rec.hold)
                    Image(systemName: "plus").font(.title2.weight(.light))
                }
                .foregroundColor(.white).frame(width: 54, height: 54).shadow(radius: 2)
                .ignoresSafeArea().allowsHitTesting(false).accessibilityHidden(true)
            }
            VStack(spacing: 10) {
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 3) {
                        Label(rec.room, systemImage: "door.left.hand.open").font(.headline)
                        Text("\(rec.frames) frames · \(rec.bytes / 1_000_000) MB")
                        Text(rec.height.map { String(format: "Height %.2f m", $0) } ?? "Finding the floor…")
                        if veil { Text("Scanned \(Int(rec.area)) m²") }
                    }
                    .font(.subheadline.monospacedDigit())
                    Spacer()
                    Button { sessions = true } label: { Image(systemName: "folder").font(.title3).frame(width: 44, height: 44) }
                        .disabled(rec.recording).accessibilityLabel("Recordings on this phone")
                }
                .card()
                if rec.tracking != "normal" {
                    Label("Tracking limited: slow down, add light, point at textured surfaces", systemImage: "exclamationmark.triangle.fill").banner(.orange)
                }
                if rec.recording { checklist }
                if let nudge {
                    Label(nudge, systemImage: "lightbulb.fill").banner(.cyan)
                } else if rec.recording && veil {
                    Text(rec.growth > 0.3 ? "New surfaces lighting up: keep sweeping slowly" : "Nothing new here: aim at the dark parts or move on")
                        .font(.callout.weight(.semibold)).padding(8).background(.black.opacity(0.5), in: Capsule())
                }
                if rec.bytes > 500_000_000 {
                    Label("Over 500 MB: stop soon (upload limit 600 MB)", systemImage: "externaldrive.fill.badge.exclamationmark").banner(.red)
                }
                Spacer()
                LazyVGrid(columns: [GridItem(), GridItem(), GridItem()]) {
                    ForEach(Self.items(zones), id: \.self) { z in
                        Button(z) { rec.enter(z) }
                            .buttonStyle(Fill(color: z == rec.room ? .accentColor : .black.opacity(0.55), height: 44))
                            .accessibilityAddTraits(z == rec.room ? .isSelected : [])
                    }
                }
                if rec.recording {
                    HStack {
                        Button { rec.mark("doorway") } label: { Label("Doorway", systemImage: "door.left.hand.closed") }
                            .buttonStyle(Fill(color: .yellow, text: .black, height: 44))
                        Button { rec.tag(); tagging = true } label: { Label("Tag", systemImage: "scope") }
                            .buttonStyle(Fill(color: .mint, text: .black, height: 44))
                    }
                }
                Button { rec.recording ? rec.stop() : rec.start() } label: {
                    Label(rec.recording ? "Stop" : "Start recording", systemImage: rec.recording ? "stop.fill" : "record.circle").font(.title2.bold())
                }
                .buttonStyle(Fill(color: rec.recording ? .red : .accentColor, height: 64))
                if let zip = rec.zip, !rec.recording {
                    HStack {
                        Button { rec.reviewing = true } label: { Label("Review", systemImage: "slider.horizontal.below.rectangle") }
                            .buttonStyle(Fill(color: .white, text: .black))
                        Button { rec.upload() } label: { Label("Upload", systemImage: "icloud.and.arrow.up") }
                            .buttonStyle(Fill()).disabled(rec.uploading)
                        Menu {
                            ShareLink(item: zip) { Label("Share / AirDrop the recording", systemImage: "square.and.arrow.up") }
                        } label: {
                            Image(systemName: "ellipsis").font(.headline).frame(width: 48, height: 48)
                                .background(.white, in: RoundedRectangle(cornerRadius: 14, style: .continuous)).foregroundColor(.black)
                        }
                        .accessibilityLabel("More")
                    }
                }
                if !rec.status.isEmpty {
                    HStack {
                        if rec.uploading { ProgressView(value: rec.sent).frame(width: 60) }
                        Text(rec.status).font(.callout)
                        Spacer()
                        if rec.uploaded { Button("Watch it build", action: showMaps).font(.callout.bold()) }
                    }
                    .card()
                }
            }
            .padding()
        }
        .environment(\.colorScheme, .dark)
        .overlay { if rec.scoring { ScoreCard(rec: rec) } }
        .onAppear { rec.run(); rec.veil.isEnabled = veil }
        .onDisappear { rec.pause() }
        .onChange(of: veil) { rec.veil.isEnabled = $0 }
        .sheet(isPresented: $sessions) { Sessions(rec: rec) }
        .sheet(isPresented: $tagging) { LabelGrid(labels: Self.items(tagLabels)) { rec.label($0); tagging = false } }
        .fullScreenCover(isPresented: $rec.reviewing) {
            Review(dir: rec.dir, zones: Self.items(zones), labels: Self.items(tagLabels)) { rec.reviewing = false; rec.save($0, $1) }
        }
    }

    var nudge: String? {
        guard rec.recording else { return nil }
        if rec.lowTilted { return "Camera straight ahead, not at the floor — like the duck's eyes" }
        if rec.lowFast { return "Slower — hold still a second in each direction" }
        if rec.sinceZone >= 8 { return "New room? Tap its name" }
        if rec.walls[rec.room, default: 0] >= 25 && rec.sectors[rec.room, default: []].count < 4 { return "Now go low ⬇︎ duck height" }
        return nil
    }

    var checklist: some View {
        let walls = rec.walls[rec.room, default: 0], low = rec.sectors[rec.room, default: []]
        let tags = rec.tags.filter { $0.room == rec.room }.count
        let tip = walls < 25 ? "Sweep every wall at chest height"
            : low.count < 8 ? "Go down to duck height: phone below your knee (under 60 cm), camera level, turn slowly all the way round"
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
        .card()
    }

    static func items(_ list: String) -> [String] {
        list.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
    }
}

/// Shown on Stop: how complete the recording is, per zone, before the review.
struct ScoreCard: View {
    @ObservedObject var rec: Recorder
    typealias Row = (mark: Color, title: String, tip: String)

    var body: some View {
        let zones = rec.walked.keys.sorted(), total = rec.walked.values.reduce(0, +)
        let short = zones.filter { rec.walked[$0, default: 0] < 4 }
        let zoneRow: Row = zones.count == 1 && total > 15 ? (.red, "1 zone for \(Int(total)) m walked", "Rooms will merge into one: fix zones in the review")
            : !short.isEmpty ? (.orange, "\(zones.count) zones, \(short.joined(separator: ", ")) under 4 m", "Check those zones in the review")
            : (.green, "\(zones.count) zones · \(Int(total)) m walked", "Each room has its own zone")
        let low: [Row] = zones.map { z in
            let k = rec.sectors[z]?.count ?? 0
            return (k >= 6 ? .green : k >= 3 ? .orange : .red, "\(z) \(k)/8 directions", k >= 6 ? "Duck height covered" : "Few low views: the duck may get lost here")
        }
        let tags: [Row] = zones.map { z in
            let k = rec.tags.filter { $0.room == z }.count
            return (k >= 2 ? .green : .orange, "\(z) \(k) tag\(k == 1 ? "" : "s")", k >= 2 ? "Enough fixed objects" : "Tag 2+ things that never move")
        }
        let mb = rec.bytes / 1_000_000, d = Int(rec.duration)
        let size: Row = (mb > 500 ? .red : .green, String(format: "%d:%02d · %d frames · %d MB", d / 60, d % 60, rec.frames, mb),
                         mb > 500 ? "Over 500 MB: the upload limit is 600 MB" : "Size is fine")
        let red = (low + [size]).filter { $0.mark == .red }.map(\.title)
        let warn = ([zoneRow] + low + tags).filter { $0.mark != .green }.map(\.title)
        let overall = !red.isEmpty ? "Consider recording again: " + red.prefix(3).joined(separator: ", ")
            : !warn.isEmpty ? "Usable, but " + warn.prefix(3).joined(separator: ", ") : "Good to upload"
        return VStack(alignment: .leading, spacing: 12) {
            Text("Recording check").font(.title2.bold())
            Text(overall).font(.headline).foregroundColor(!red.isEmpty ? .red : !warn.isEmpty ? .orange : .green)
            ScrollView {
                VStack(alignment: .leading, spacing: 8) {
                    section("Zones", [zoneRow])
                    section("Duck-height pass", low)
                    section("Tags", tags)
                    section("Length", [size])
                }
            }
            HStack {
                Button("Skip") { rec.scoring = false; rec.save(nil, nil) }.buttonStyle(Fill(color: .gray.opacity(0.4)))
                Button("Review zones & tags") { rec.scoring = false; rec.reviewing = true }.buttonStyle(Fill())
            }
        }
        .foregroundColor(.white).padding()
        .background(Color.black.ignoresSafeArea())
    }

    func section(_ name: String, _ rows: [Row]) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(name.uppercased()).font(.caption.bold()).foregroundColor(.gray)
            ForEach(rows.indices, id: \.self) { i in
                HStack(alignment: .firstTextBaseline) {
                    Circle().fill(rows[i].mark).frame(width: 12, height: 12)
                    VStack(alignment: .leading) {
                        Text(rows[i].title).font(.body.bold().monospacedDigit())
                        Text(rows[i].tip).font(.caption).foregroundColor(.white.opacity(0.75))
                    }
                }
            }
        }
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
                    Button(l) { pick(l) }.buttonStyle(Fill(height: 44))
                }
            }
            Button("Cancel") { pick(nil) }.frame(maxWidth: .infinity, minHeight: 44)
        }
        .padding().presentationDetents([.medium]).interactiveDismissDisabled()
    }
}

struct Settings: View {
    @Binding var zones: String
    @Binding var veil: Bool
    @Binding var tagLabels: String
    @EnvironmentObject var auth: Auth
    @AppStorage("user") var user = ""
    let version = Bundle.main.infoDictionary.map { "\($0["CFBundleShortVersionString"] ?? "?") (\($0["CFBundleVersion"] ?? "?"))" } ?? "?"

    var body: some View {
        NavigationStack {
            Form {
                Section("Account") {
                    LabeledContent("Signed in as", value: user.isEmpty ? "Access link" : user)
                    LabeledContent("Server", value: Auth.server.host ?? "")
                    Button(role: .destructive) { auth.signOut() } label: { Label("Sign out", systemImage: "rectangle.portrait.and.arrow.right") }
                }
                Section(header: Text("Zones"), footer: Text("Comma-separated. One button per room you walk through.")) {
                    TextField("Kitchen, Living room, …", text: $zones)
                }
                Section(header: Text("Tag labels"), footer: Text("Comma-separated. Things that never move, tagged while recording.")) {
                    TextField("Fridge, Oven, …", text: $tagLabels)
                }
                Section(footer: Text("Dark = not scanned yet; surfaces light up once the LiDAR mesh covers them.")) {
                    Toggle("Coverage veil", isOn: $veil)
                }
                Section { LabeledContent("Version", value: version) }
            }
            .navigationTitle("Settings")
        }
    }
}

struct Sessions: View {
    @ObservedObject var rec: Recorder
    @State var ids: [String] = []
    @Environment(\.dismiss) var dismiss

    var body: some View {
        NavigationStack {
            List {
                ForEach(ids, id: \.self) { id in
                    Button { rec.open(id); dismiss() } label: { Label(sessionTitle(id), systemImage: "video.fill") }
                }
                .onDelete { $0.forEach { rec.delete(ids[$0]) }; ids = rec.sessions() }
            }
            .overlay {
                if ids.isEmpty { Empty(icon: "video.slash", title: "No recordings yet", text: "Tap Start recording to walk through your home.") }
            }
            .navigationTitle("Recordings")
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
                Button { cursor = max(cursor - 1, 0) } label: { Image(systemName: "chevron.left").frame(width: 44, height: 44) }
                    .accessibilityLabel("Previous frame")
                Button(anchor == nil ? "Select range from here" : "Cancel range") { anchor = anchor == nil ? cursor : nil }
                    .buttonStyle(Fill(color: anchor == nil ? .gray.opacity(0.3) : .white, text: anchor == nil ? .white : .black, height: 44))
                Button { cursor = min(cursor + 1, max(rooms.count - 1, 0)) } label: { Image(systemName: "chevron.right").frame(width: 44, height: 44) }
                    .accessibilityLabel("Next frame")
            }
            Text(anchor == nil ? "Tap a zone: it applies from here to the next change" : "Tap a zone: it applies to the selected range")
                .font(.caption).foregroundColor(.white.opacity(0.8))
            LazyVGrid(columns: [GridItem(), GridItem(), GridItem()]) {
                ForEach(zones, id: \.self) { z in
                    Button(z) { assign(z) }.buttonStyle(Fill(color: color(z), height: 44))
                }
            }
            HStack {
                Button("Undo all") { rooms = orig; anchor = nil }.buttonStyle(Fill(color: .gray.opacity(0.4)))
                Button("Skip") { if tags.isEmpty { done(nil, nil) } else { keep = nil; tagStep = true } }.buttonStyle(Fill(color: .gray.opacity(0.4)))
                Button(tags.isEmpty ? "Save" : "Next") { if tags.isEmpty { done(rooms, []) } else { keep = rooms; tagStep = true } }.buttonStyle(Fill())
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
            .overlay { if tags.isEmpty { Empty(icon: "scope", title: "No tags left", text: "Save to build without tags, or go back.") } }
            HStack {
                Button("Back") { tagStep = false }.buttonStyle(Fill(color: .gray.opacity(0.4)))
                Button("Save") { done(keep, tags) }.buttonStyle(Fill())
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
