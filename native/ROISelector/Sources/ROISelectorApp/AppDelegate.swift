import AppKit
import NPYCore
import ROISelectorCore

final class AppDelegate: NSObject, NSApplicationDelegate {
    private let configuration: AppConfiguration
    private var windowController: MainWindowController?

    init(configuration: AppConfiguration) {
        self.configuration = configuration
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        buildMenu()

        do {
            let array = try NPYArray(contentsOf: configuration.sourceURL)
            let displayImage = try NPYDisplayImage(array: array)
            let document = try TemplateDocument.loadOrCreate(
                sourceURL: configuration.sourceURL,
                templateURL: configuration.templateURL,
                imageWidth: array.width,
                imageHeight: array.height,
                groupsSpec: configuration.groupsSpec,
                elementsSpec: configuration.elementsSpec
            )

            let controller = MainWindowController(
                document: document,
                displayImage: displayImage
            )
            controller.showWindow(nil)
            windowController = controller

            NSApp.setActivationPolicy(.regular)
            NSApp.activate(ignoringOtherApps: true)
        } catch {
            showFatalError(error)
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    @objc func saveDocument(_ sender: Any?) {
        windowController?.saveDocument()
    }

    @objc func selectNextROI(_ sender: Any?) {
        windowController?.selectNextROI()
    }

    @objc func selectPreviousROI(_ sender: Any?) {
        windowController?.selectPreviousROI()
    }

    @objc func clearCurrentROI(_ sender: Any?) {
        windowController?.clearCurrentROI()
    }

    @objc func resetZoom(_ sender: Any?) {
        windowController?.resetZoom()
    }

    private func buildMenu() {
        let mainMenu = NSMenu()

        let appMenuItem = NSMenuItem()
        mainMenu.addItem(appMenuItem)
        let appMenu = NSMenu()
        appMenu.addItem(
            withTitle: "Quit ROISelector",
            action: #selector(NSApplication.terminate(_:)),
            keyEquivalent: "q"
        )
        appMenuItem.submenu = appMenu

        let fileMenuItem = NSMenuItem()
        mainMenu.addItem(fileMenuItem)
        let fileMenu = NSMenu(title: "File")
        fileMenu.addItem(
            withTitle: "Save",
            action: #selector(saveDocument(_:)),
            keyEquivalent: "s"
        )
        fileMenuItem.submenu = fileMenu

        let editMenuItem = NSMenuItem()
        mainMenu.addItem(editMenuItem)
        let editMenu = NSMenu(title: "Edit")
        editMenu.addItem(
            withTitle: "Clear Current ROI",
            action: #selector(clearCurrentROI(_:)),
            keyEquivalent: "\u{8}"
        )
        editMenuItem.submenu = editMenu

        let roiMenuItem = NSMenuItem()
        mainMenu.addItem(roiMenuItem)
        let roiMenu = NSMenu(title: "ROI")
        roiMenu.addItem(
            withTitle: "Previous ROI",
            action: #selector(selectPreviousROI(_:)),
            keyEquivalent: "["
        )
        roiMenu.addItem(
            withTitle: "Next ROI",
            action: #selector(selectNextROI(_:)),
            keyEquivalent: "]"
        )
        roiMenuItem.submenu = roiMenu

        let viewMenuItem = NSMenuItem()
        mainMenu.addItem(viewMenuItem)
        let viewMenu = NSMenu(title: "View")
        viewMenu.addItem(
            withTitle: "Reset Zoom",
            action: #selector(resetZoom(_:)),
            keyEquivalent: "0"
        )
        viewMenuItem.submenu = viewMenu

        NSApp.mainMenu = mainMenu
    }

    private func showFatalError(_ error: Error) {
        let alert = NSAlert(error: error)
        alert.messageText = "Could Not Open ROI Selector"
        alert.runModal()
        NSApp.terminate(nil)
    }
}
