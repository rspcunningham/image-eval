import AppKit
import ROISelectorCore

final class MainViewController: NSViewController {
    private let document: TemplateDocument
    private let displayImage: NPYDisplayImage
    private let sidebarController = ROIListViewController()
    private let canvasView = ROICanvasView()
    private let divider = NSBox()
    private var activeID: ROIIdentifier

    init(document: TemplateDocument, displayImage: NPYDisplayImage) {
        self.document = document
        self.displayImage = displayImage
        let entries = document.entries()
        self.activeID = entries.first { !$0.isComplete }?.id
            ?? entries.first?.id
            ?? .normalization(.black)
        super.init(nibName: nil, bundle: nil)
    }

    required init?(coder: NSCoder) {
        nil
    }

    override func loadView() {
        view = NSView()
        view.wantsLayer = true
        view.layer?.backgroundColor = NSColor.black.cgColor

        addChild(sidebarController)
        let sidebarView = sidebarController.view
        sidebarView.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(sidebarView)

        divider.boxType = .custom
        divider.fillColor = NSColor.separatorColor
        divider.borderColor = .clear
        divider.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(divider)

        canvasView.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(canvasView)

        NSLayoutConstraint.activate([
            sidebarView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            sidebarView.topAnchor.constraint(equalTo: view.topAnchor),
            sidebarView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            sidebarView.widthAnchor.constraint(equalToConstant: 280),

            divider.leadingAnchor.constraint(equalTo: sidebarView.trailingAnchor),
            divider.topAnchor.constraint(equalTo: view.topAnchor),
            divider.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            divider.widthAnchor.constraint(equalToConstant: 1),

            canvasView.leadingAnchor.constraint(equalTo: divider.trailingAnchor),
            canvasView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            canvasView.topAnchor.constraint(equalTo: view.topAnchor),
            canvasView.bottomAnchor.constraint(equalTo: view.bottomAnchor)
        ])
    }

    override func viewDidLoad() {
        super.viewDidLoad()

        sidebarController.onSelectionChanged = { [weak self] id in
            self?.selectROI(id)
        }
        sidebarController.onNext = { [weak self] in
            self?.selectNextROI()
        }
        sidebarController.onPrevious = { [weak self] in
            self?.selectPreviousROI()
        }
        sidebarController.onClear = { [weak self] in
            self?.clearCurrentROI()
        }
        sidebarController.onZoomChanged = { [weak self] zoom in
            self?.canvasView.setZoom(zoom)
        }
        sidebarController.onResetZoom = { [weak self] in
            self?.resetZoom()
        }

        canvasView.onRectChanged = { [weak self] rect in
            self?.setActiveRect(rect)
        }
        canvasView.onAdvanceRequested = { [weak self] in
            self?.selectNextROI()
        }
        canvasView.onClearRequested = { [weak self] in
            self?.clearCurrentROI()
        }
        canvasView.onZoomChanged = { [weak self] zoom in
            self?.sidebarController.setZoom(zoom)
        }

        refreshViews()
    }

    override func viewDidAppear() {
        super.viewDidAppear()
        view.window?.makeFirstResponder(canvasView)
    }

    func saveDocument() {
        do {
            try document.save()
            refreshViews()
        } catch {
            showError(error)
        }
    }

    func selectNextROI() {
        let entries = document.entries()
        guard
            let index = entries.firstIndex(where: { $0.id == activeID }),
            index + 1 < entries.count
        else {
            return
        }
        selectROI(entries[index + 1].id)
    }

    func selectPreviousROI() {
        let entries = document.entries()
        guard
            let index = entries.firstIndex(where: { $0.id == activeID }),
            index > 0
        else {
            return
        }
        selectROI(entries[index - 1].id)
    }

    func clearCurrentROI() {
        setActiveRect(nil)
    }

    func resetZoom() {
        canvasView.resetView()
    }

    private func selectROI(_ id: ROIIdentifier) {
        activeID = id
        refreshViews()
        view.window?.makeFirstResponder(canvasView)
    }

    private func setActiveRect(_ rect: PixelRect?) {
        do {
            try document.setRect(rect, for: activeID)
            refreshViews()
        } catch {
            showError(error)
        }
    }

    private func refreshViews() {
        let entries = document.entries()
        sidebarController.reload(entries: entries, selectedID: activeID)
        canvasView.configure(
            displayImage: displayImage,
            entries: entries,
            activeID: activeID
        )
        sidebarController.setZoom(canvasView.currentZoom)
    }

    private func showError(_ error: Error) {
        let alert = NSAlert(error: error)
        alert.runModal()
    }
}
