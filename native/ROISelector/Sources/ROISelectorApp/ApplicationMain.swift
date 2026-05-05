import AppKit
import ROISelectorCore

private var retainedDelegate: AppDelegate?

public func runROISelectorApplication(configuration: AppConfiguration) {
    let application = NSApplication.shared
    let delegate = AppDelegate(configuration: configuration)
    retainedDelegate = delegate
    application.delegate = delegate
    application.run()
}
