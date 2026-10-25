import Security
import SwiftUI

/// The sign-in token (or a pasted access key) lives in the Keychain, this device only.
enum Keychain {
    private static let item: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
                                              kSecAttrService as String: "io.housetwinpatrol.lidar.token"]

    static func read() -> String? {
        var out: AnyObject?
        guard SecItemCopyMatching(item.merging([kSecReturnData as String: true]) { $1 } as CFDictionary, &out) == errSecSuccess,
              let data = out as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    static func save(_ value: String?) {
        SecItemDelete(item as CFDictionary)
        guard let value else { return }
        SecItemAdd(item.merging([kSecValueData as String: Data(value.utf8),
                                 kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly]) { $1 } as CFDictionary, nil)
    }
}

enum ApiError: LocalizedError {
    case offline, unauthorized, server(Int, String)

    var errorDescription: String? {
        switch self {
        case .offline: return "You're offline. Check your connection and try again."
        case .unauthorized: return "Your session has ended. Please sign in again."
        case .server(401, _): return "Wrong user name or password."
        case .server(let code, let detail): return detail.isEmpty ? "The server had a problem (\(code)). Try again in a minute." : detail.prefix(1).uppercased() + detail.dropFirst() + "."
        }
    }

    static func from(_ error: Error) -> Error {
        let offline: [URLError.Code] = [.notConnectedToInternet, .networkConnectionLost, .timedOut, .cannotFindHost, .cannotConnectToHost, .dataNotAllowed]
        if let e = error as? URLError, offline.contains(e.code) { return ApiError.offline }
        return error
    }
}

/// The server, the token and every request to it. A 401 anywhere signs out, which shows the login screen.
final class Auth: ObservableObject {
    static let shared = Auth()
    static let server = URL(string: Bundle.main.object(forInfoDictionaryKey: "ServerURL") as? String ?? "https://game.adtomic.co")!
    @Published private(set) var token = Keychain.read()

    init() {
        guard !UserDefaults.standard.bool(forKey: "installed") else { return }
        UserDefaults.standard.set(true, forKey: "installed")
        set(nil)  // the Keychain outlives a reinstall: start signed out
    }

    func set(_ token: String?) {
        Keychain.save(token)
        self.token = token
    }

    func signOut() { set(nil) }

    func url(_ path: String, _ query: [String: String] = [:], key: Bool = false) -> URL {
        var c = URLComponents(url: Self.server.appendingPathComponent(path), resolvingAgainstBaseURL: false)!
        let items = query.sorted { $0.key < $1.key }.map { URLQueryItem(name: $0.key, value: $0.value) }
        c.queryItems = items + (key ? [URLQueryItem(name: "k", value: token)] : [])  // the web pages read ?k=
        return c.url!
    }

    func request(_ path: String, _ query: [String: String] = [:], method: String = "GET", body: Data? = nil) -> URLRequest {
        var r = URLRequest(url: url(path, query), timeoutInterval: 30)
        r.httpMethod = method
        r.httpBody = body
        if body != nil { r.setValue("application/json", forHTTPHeaderField: "Content-Type") }
        if let token { r.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        return r
    }

    @MainActor func data(_ path: String, _ query: [String: String] = [:], method: String = "GET", body: Data? = nil) async throws -> Data {
        let data: Data, resp: URLResponse
        do { (data, resp) = try await URLSession.shared.data(for: request(path, query, method: method, body: body)) } catch { throw ApiError.from(error) }
        let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
        if code == 401 && token != nil { signOut(); throw ApiError.unauthorized }
        guard code == 200 else {
            let detail = (try? JSONSerialization.jsonObject(with: data) as? [String: Any])?["detail"] as? String
            throw ApiError.server(code, detail ?? "")
        }
        return data
    }

    @MainActor func json<T: Decodable>(_ path: String, _ query: [String: String] = [:], method: String = "GET", body: Data? = nil) async throws -> T {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        return try d.decode(T.self, from: await data(path, query, method: method, body: body))
    }

    @MainActor func login(user: String, password: String) async throws {
        struct Reply: Decodable { let token: String }
        let reply: Reply = try await json("/auth/login", method: "POST", body: try JSONEncoder().encode(["user": user, "password": password]))
        set(reply.token)
    }
}

struct LoginView: View {
    @EnvironmentObject var auth: Auth
    @AppStorage("user") var user = ""
    @State var password = ""
    @State var reveal = false
    @State var busy = false
    @State var error: String?
    @State var link = ""
    @FocusState var focus: Int?

    var body: some View {
        ScrollView {
            VStack(spacing: 28) {
                VStack(spacing: 12) {
                    Image("LaunchLogo").resizable().frame(width: 88, height: 88).accessibilityHidden(true)
                    Text("Twin LiDAR").font(.largeTitle.bold())
                    Text("Walk through your home once. Get its floor plan and a 3D twin.")
                        .font(.callout).foregroundStyle(.secondary).multilineTextAlignment(.center)
                }
                .padding(.top, 48)
                VStack(spacing: 12) {
                    field("person.fill") {
                        TextField("User name", text: $user)
                            .textContentType(.username).textInputAutocapitalization(.never).autocorrectionDisabled()
                            .focused($focus, equals: 0).submitLabel(.next).onSubmit { focus = 1 }
                    }
                    field("lock.fill") {
                        Group {
                            if reveal { TextField("Password", text: $password) } else { SecureField("Password", text: $password) }
                        }
                        .textContentType(.password).textInputAutocapitalization(.never).autocorrectionDisabled()
                        .focused($focus, equals: 1).submitLabel(.go).onSubmit { Task { await signIn() } }
                        Button { reveal.toggle() } label: { Image(systemName: reveal ? "eye.slash" : "eye").foregroundStyle(.secondary) }
                            .accessibilityLabel(reveal ? "Hide password" : "Show password")
                    }
                }
                if let error {
                    Label(error, systemImage: "exclamationmark.triangle.fill").font(.callout).foregroundStyle(.red)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                Button { Task { await signIn() } } label: {
                    if busy { ProgressView().tint(.white) } else { Text("Sign in") }
                }
                .buttonStyle(Fill()).disabled(user.isEmpty || password.isEmpty || busy)
                DisclosureGroup("Use an access link instead") {
                    TextField("https://…/?k=…", text: $link)
                        .textInputAutocapitalization(.never).autocorrectionDisabled().textFieldStyle(.roundedBorder).padding(.top, 6)
                        .onChange(of: link) { v in
                            let k = URLComponents(string: v.trimmingCharacters(in: .whitespacesAndNewlines))?.queryItems?.first { $0.name == "k" }?.value
                            if let k, !k.isEmpty { user = ""; auth.set(k) }
                        }
                }
                .font(.footnote).tint(.secondary)
            }
            .padding(24).frame(maxWidth: 440)
            .frame(maxWidth: .infinity)
        }
        .background(Color(.systemGroupedBackground).ignoresSafeArea())
        .onAppear { focus = user.isEmpty ? 0 : 1 }
    }

    func field<Content: View>(_ icon: String, @ViewBuilder _ content: () -> Content) -> some View {
        HStack(spacing: 12) {
            Image(systemName: icon).foregroundStyle(.secondary).frame(width: 22).accessibilityHidden(true)
            content()
        }
        .padding(.horizontal, 14).frame(minHeight: 52)
        .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }

    func signIn() async {
        guard !user.isEmpty, !password.isEmpty, !busy else { return }
        busy = true
        error = nil
        do { try await auth.login(user: user.trimmingCharacters(in: .whitespaces), password: password) } catch { self.error = error.localizedDescription }
        busy = false
    }
}
