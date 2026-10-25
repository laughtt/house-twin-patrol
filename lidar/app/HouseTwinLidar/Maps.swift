import SwiftUI
import UserNotifications
import WebKit

/// "lidar-20261025-153000" → "25 Oct 2026 at 15:30".
func sessionTitle(_ id: String) -> String {
    let f = DateFormatter()
    f.dateFormat = "'lidar-'yyyyMMdd-HHmmss"
    return f.date(from: id)?.formatted(date: .abbreviated, time: .shortened) ?? id
}

struct MapItem: Decodable, Identifiable {
    let session: String
    let state: String
    var rooms: Int?
    var areaM2: Double?
    var id: String { session }

    var icon: (name: String, color: Color) {
        switch state {
        case "done": return ("checkmark.seal.fill", .green)
        case "building": return ("hammer.fill", .accentColor)
        case "failed": return ("exclamationmark.triangle.fill", .orange)
        default: return ("clock.fill", .secondary)
        }
    }

    var summary: String {
        switch state {
        case "done": return ["Ready", rooms.map { "\($0) room\($0 == 1 ? "" : "s")" }, areaM2.map { String(format: "%.0f m²", $0) }].compactMap { $0 }.joined(separator: " · ")
        case "building": return "Building your map…"
        case "failed": return "Build failed"
        default: return "Uploaded, waiting to build"
        }
    }
}

/// The server's maps, polled every 5 s while one builds; a local notification when one turns ready.
final class Maps: NSObject, ObservableObject, UNUserNotificationCenterDelegate {
    @Published var items: [MapItem] = []
    @Published var error: String?
    @Published var loaded = false
    var busy: Bool { items.contains { $0.state == "building" } }

    override init() {
        super.init()
        UNUserNotificationCenter.current().delegate = self
    }

    @MainActor func refresh() async {
        guard Auth.shared.token != nil else { return }
        do {
            let new: [MapItem] = try await Auth.shared.json("/lidar/maps")
            let was = Dictionary(items.map { ($0.session, $0.state) }) { a, _ in a }
            for m in new where m.state == "done" && was[m.session] == "building" { notify(m) }
            items = new
            error = nil
        } catch {
            if (error as? URLError)?.code != .cancelled { self.error = error.localizedDescription }
        }
        loaded = true
    }

    func askToNotify() {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound]) { _, _ in }
    }

    private func notify(_ m: MapItem) {
        let c = UNMutableNotificationContent()
        c.title = "Your map is ready"
        c.body = "\(sessionTitle(m.session)): open My maps to see the floor plan and 3D."
        c.sound = .default
        UNUserNotificationCenter.current().add(UNNotificationRequest(identifier: m.session, content: c, trigger: nil))
    }

    func userNotificationCenter(_ center: UNUserNotificationCenter, willPresent notification: UNNotification,
                                withCompletionHandler done: @escaping (UNNotificationPresentationOptions) -> Void) {
        done([.banner, .sound])
    }
}

struct MapsView: View {
    @ObservedObject var maps: Maps
    @ObservedObject var rec: Recorder

    var body: some View {
        NavigationStack {
            List {
                if rec.uploading {
                    Section {
                        VStack(alignment: .leading, spacing: 8) {
                            Label("Uploading \(sessionTitle(rec.sessionID))", systemImage: "icloud.and.arrow.up").font(.headline)
                            ProgressView(value: rec.sent)
                            Text("\(Int(rec.sent * 100)) % · keep the app open").font(.caption).foregroundStyle(.secondary)
                        }
                        .padding(.vertical, 4)
                    }
                }
                if let error = maps.error, !maps.items.isEmpty {
                    Label(error, systemImage: "wifi.exclamationmark").font(.callout).foregroundStyle(.orange)
                }
                ForEach(maps.items) { m in
                    NavigationLink(value: m.session) {
                        HStack(spacing: 14) {
                            Group {
                                if m.state == "building" { ProgressView() } else { Image(systemName: m.icon.name).foregroundStyle(m.icon.color) }
                            }
                            .font(.title2).frame(width: 34)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(sessionTitle(m.session)).font(.headline)
                                Text(m.summary).font(.subheadline).foregroundStyle(.secondary)
                            }
                        }
                        .padding(.vertical, 4)
                        .accessibilityElement(children: .combine)
                    }
                }
            }
            .listStyle(.insetGrouped)
            .navigationTitle("My maps")
            .navigationDestination(for: String.self) { MapDetail(session: $0, maps: maps) }
            .refreshable { await maps.refresh() }
            .overlay {
                if maps.items.isEmpty && !rec.uploading {
                    if !maps.loaded { ProgressView("Loading your maps…") }
                    else if let error = maps.error {
                        Empty(icon: "wifi.exclamationmark", title: "Can't reach the server", text: error) { Task { await maps.refresh() } }
                    } else {
                        Empty(icon: "map", title: "No maps yet",
                              text: "Record a walk through your home and upload it. Its floor plan and 3D model show up here when they're ready.")
                    }
                }
            }
        }
    }
}

struct MapDetail: View {
    let session: String
    @ObservedObject var maps: Maps
    @State var tab = 0
    @State var plan: UIImage?
    @State var status: Status?
    @State var error: String?
    @State var rebuilding = false
    var item: MapItem? { maps.items.first { $0.session == session } }

    struct Status: Decodable {
        struct Space: Decodable { let wM: Double, lM: Double, areaM2: Double }
        struct Report: Decodable { let spaces: [String: Space], doors: Int?, objects: Int?, landmarks: Int? }
        let report: Report?
        let log: String?
    }

    var body: some View {
        Group {
            if item?.state == "done" {
                VStack(spacing: 0) {
                    Picker("View", selection: $tab) {
                        Text("Floor plan").tag(0)
                        Text("3D").tag(1)
                    }
                    .pickerStyle(.segmented).padding(.horizontal).padding(.bottom, 8)
                    if tab == 0 { planView } else { WebView(url: Auth.shared.url("/lidar/3d", ["s": session], key: true)).ignoresSafeArea(edges: .bottom) }
                }
            } else {
                ScrollView { notReady }
            }
        }
        .background(Color(.systemGroupedBackground))
        .navigationTitle(sessionTitle(session)).navigationBarTitleDisplayMode(.inline)
        .toolbar {
            if let plan, tab == 0 {
                ShareLink(item: Image(uiImage: plan), preview: SharePreview("Floor plan", image: Image(uiImage: plan)))
            }
        }
        .task(id: item?.state) { await load() }
    }

    var planView: some View {
        VStack(spacing: 12) {
            ZStack {
                if let plan { ZoomImage(image: plan) }
                else if let error { Empty(icon: "photo", title: "No floor plan", text: error) { Task { await load() } } }
                else { ProgressView("Loading the floor plan…") }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
            if let r = status?.report {
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(r.spaces.sorted { $0.key < $1.key }, id: \.key) { name, s in
                        HStack {
                            Text(name).font(.subheadline.weight(.semibold))
                            Spacer()
                            Text(String(format: "%.1f × %.1f m · %.0f m²", s.wM, s.lM, s.areaM2)).font(.subheadline.monospacedDigit()).foregroundStyle(.secondary)
                        }
                    }
                    Text([r.doors.map { "\($0) doors" }, r.objects.map { "\($0) objects" }, r.landmarks.map { "\($0) landmarks" }].compactMap { $0 }.joined(separator: " · "))
                        .font(.caption).foregroundStyle(.secondary)
                }
                .padding(14)
                .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
            }
        }
        .padding([.horizontal, .bottom])
    }

    var notReady: some View {
        VStack(spacing: 16) {
            switch item?.state {
            case "building":
                ProgressView().controlSize(.large).padding(.top, 60)
                Text("Building your map").font(.title3.bold())
                Text("This takes a few minutes. You can leave this screen: you'll get a notification when it's ready.")
                    .font(.callout).foregroundStyle(.secondary).multilineTextAlignment(.center)
            case "failed", "not built":
                Empty(icon: item?.icon.name ?? "clock.fill", title: item?.state == "failed" ? "The build failed" : "Not built yet",
                      text: item?.state == "failed" ? "Try building it again. If it fails again, record the walk again with slower sweeps and a zone per room."
                                                    : "The server was busy when this walk arrived. Build it now.")
                Button { Task { await rebuild() } } label: { Label("Build again", systemImage: "hammer") }
                    .buttonStyle(Fill()).disabled(rebuilding).frame(maxWidth: 320)
                if let log = status?.log, !log.isEmpty {
                    DisclosureGroup("Details") { Text(log.suffix(600)).font(.caption.monospaced()).frame(maxWidth: .infinity, alignment: .leading) }
                        .font(.footnote)
                }
            default:
                Empty(icon: "questionmark.folder", title: "Map not found", text: "Pull to refresh My maps.")
            }
            if let error { Label(error, systemImage: "exclamationmark.triangle.fill").font(.callout).foregroundStyle(.orange) }
        }
        .padding(24)
    }

    func load() async {
        guard let state = item?.state, state != "building" else { return }
        do {
            status = try await Auth.shared.json("/lidar/status", ["session": session])
            if state == "done", plan == nil {
                plan = UIImage(data: try await Auth.shared.data("/lidar/file", ["session": session, "name": "top.png"]))
            }
            error = nil
        } catch {
            if (error as? URLError)?.code != .cancelled { self.error = error.localizedDescription }
        }
    }

    func rebuild() async {
        rebuilding = true
        do {
            _ = try await Auth.shared.data("/lidar/build", ["session": session], method: "POST")
            error = nil
            await maps.refresh()
        } catch { self.error = error.localizedDescription }
        rebuilding = false
    }
}

/// Pinch to zoom, drag to pan.
struct ZoomImage: UIViewRepresentable {
    let image: UIImage

    func makeUIView(context: Context) -> UIScrollView {
        let s = UIScrollView(), v = context.coordinator.view
        v.image = image
        v.contentMode = .scaleAspectFit
        v.isAccessibilityElement = true
        v.accessibilityLabel = "Floor plan"
        v.translatesAutoresizingMaskIntoConstraints = false
        s.addSubview(v)
        NSLayoutConstraint.activate([
            v.widthAnchor.constraint(equalTo: s.frameLayoutGuide.widthAnchor), v.heightAnchor.constraint(equalTo: s.frameLayoutGuide.heightAnchor),
            v.leadingAnchor.constraint(equalTo: s.contentLayoutGuide.leadingAnchor), v.trailingAnchor.constraint(equalTo: s.contentLayoutGuide.trailingAnchor),
            v.topAnchor.constraint(equalTo: s.contentLayoutGuide.topAnchor), v.bottomAnchor.constraint(equalTo: s.contentLayoutGuide.bottomAnchor)])
        s.maximumZoomScale = 8
        s.delegate = context.coordinator
        return s
    }

    func updateUIView(_ s: UIScrollView, context: Context) { context.coordinator.view.image = image }
    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator: NSObject, UIScrollViewDelegate {
        let view = UIImageView()
        func viewForZooming(in scrollView: UIScrollView) -> UIView? { view }
    }
}

struct WebView: UIViewRepresentable {
    let url: URL

    func makeUIView(context: Context) -> WKWebView {
        let w = WKWebView()
        w.isOpaque = false
        w.backgroundColor = .black
        w.load(URLRequest(url: url))
        return w
    }

    func updateUIView(_ w: WKWebView, context: Context) {}
}
