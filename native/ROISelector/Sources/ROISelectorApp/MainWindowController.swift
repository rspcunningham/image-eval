import AppKit
import ROISelectorCore

final class MainWindowController: NSWindowController {
    private let mainViewController: MainViewController

    init(document: TemplateDocument, displayImage: NPYDisplayImage) {
        self.mainViewController = MainViewController(
            document: document,
            displayImage: displayImage
        )

        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1220, height: 860),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "ROI Selector - \(document.url.lastPathComponent)"
        window.contentMinSize = NSSize(width: 960, height: 640)
        window.isRestorable = false
        window.contentViewController = mainViewController

        super.init(window: window)
        window.center()
    }

    required init?(coder: NSCoder) {
        nil
    }

    func saveDocument() {
        mainViewController.saveDocument()
    }

    func selectNextROI() {
        mainViewController.selectNextROI()
    }

    func selectPreviousROI() {
        mainViewController.selectPreviousROI()
    }

    func clearCurrentROI() {
        mainViewController.clearCurrentROI()
    }

    func resetZoom() {
        mainViewController.resetZoom()
    }
}
