import AppKit
import ROISelectorCore

final class ROIListViewController: NSViewController, NSTableViewDataSource, NSTableViewDelegate {
    private static let sliderMinZoom = Double(ROICanvasView.minimumZoom)
    private static let sliderMaxZoom = 32.0

    private let titleLabel = NSTextField(labelWithString: "ROIs")
    private let statusLabel = NSTextField(labelWithString: "")
    private let scrollView = NSScrollView()
    private let tableView = NSTableView()
    private let zoomLabel = NSTextField(labelWithString: "Zoom")
    private let zoomValueLabel = NSTextField(labelWithString: "")
    private let zoomSlider = NSSlider(value: 0, minValue: 0, maxValue: 1, target: nil, action: nil)
    private let zoomFitButton = NSButton(title: "Fit", target: nil, action: nil)
    private let previousButton = NSButton(title: "Previous", target: nil, action: nil)
    private let nextButton = NSButton(title: "Next", target: nil, action: nil)
    private let clearButton = NSButton(title: "Clear", target: nil, action: nil)

    private var entries: [ROIListEntry] = []
    private var selectedID: ROIIdentifier?
    private var isProgrammaticSelection = false
    private var isUpdatingZoomControl = false

    var onSelectionChanged: ((ROIIdentifier) -> Void)?
    var onNext: (() -> Void)?
    var onPrevious: (() -> Void)?
    var onClear: (() -> Void)?
    var onZoomChanged: ((CGFloat) -> Void)?
    var onResetZoom: (() -> Void)?

    override func loadView() {
        view = NSView()
        view.wantsLayer = true
        view.layer?.backgroundColor = NSColor.controlBackgroundColor.cgColor

        titleLabel.font = .boldSystemFont(ofSize: 17)
        titleLabel.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(titleLabel)

        statusLabel.font = .systemFont(ofSize: 12)
        statusLabel.textColor = .secondaryLabelColor
        statusLabel.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(statusLabel)

        let column = NSTableColumn(identifier: NSUserInterfaceItemIdentifier("roi"))
        column.title = "ROI"
        tableView.addTableColumn(column)
        tableView.headerView = nil
        tableView.rowHeight = 28
        tableView.intercellSpacing = NSSize(width: 0, height: 2)
        tableView.dataSource = self
        tableView.delegate = self
        tableView.allowsEmptySelection = false
        tableView.backgroundColor = .clear

        scrollView.documentView = tableView
        scrollView.hasVerticalScroller = true
        scrollView.borderType = .noBorder
        scrollView.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(scrollView)

        zoomLabel.font = .systemFont(ofSize: 12, weight: .medium)
        zoomLabel.textColor = .secondaryLabelColor
        zoomValueLabel.font = .monospacedDigitSystemFont(ofSize: 12, weight: .regular)
        zoomValueLabel.textColor = .secondaryLabelColor
        zoomValueLabel.alignment = .right

        zoomSlider.target = self
        zoomSlider.action = #selector(zoomSliderChanged(_:))
        zoomSlider.isContinuous = true

        zoomFitButton.target = self
        zoomFitButton.action = #selector(zoomFitPressed(_:))
        zoomFitButton.setContentHuggingPriority(.required, for: .horizontal)

        let zoomHeaderStack = NSStackView(views: [zoomLabel, zoomValueLabel])
        zoomHeaderStack.orientation = .horizontal
        zoomHeaderStack.distribution = .fill
        zoomHeaderStack.spacing = 8

        let zoomSliderStack = NSStackView(views: [zoomSlider, zoomFitButton])
        zoomSliderStack.orientation = .horizontal
        zoomSliderStack.distribution = .fill
        zoomSliderStack.spacing = 8

        let zoomStack = NSStackView(views: [zoomHeaderStack, zoomSliderStack])
        zoomStack.orientation = .vertical
        zoomStack.spacing = 6
        zoomStack.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(zoomStack)

        previousButton.target = self
        previousButton.action = #selector(previousPressed(_:))
        nextButton.target = self
        nextButton.action = #selector(nextPressed(_:))
        clearButton.target = self
        clearButton.action = #selector(clearPressed(_:))

        let buttonStack = NSStackView(views: [previousButton, nextButton, clearButton])
        buttonStack.orientation = .horizontal
        buttonStack.distribution = .fillEqually
        buttonStack.spacing = 8
        buttonStack.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(buttonStack)

        NSLayoutConstraint.activate([
            titleLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 16),
            titleLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -16),
            titleLabel.topAnchor.constraint(equalTo: view.topAnchor, constant: 16),

            statusLabel.leadingAnchor.constraint(equalTo: titleLabel.leadingAnchor),
            statusLabel.trailingAnchor.constraint(equalTo: titleLabel.trailingAnchor),
            statusLabel.topAnchor.constraint(equalTo: titleLabel.bottomAnchor, constant: 4),

            scrollView.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 8),
            scrollView.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -8),
            scrollView.topAnchor.constraint(equalTo: statusLabel.bottomAnchor, constant: 12),
            scrollView.bottomAnchor.constraint(equalTo: zoomStack.topAnchor, constant: -12),

            zoomStack.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 12),
            zoomStack.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -12),
            zoomStack.bottomAnchor.constraint(equalTo: buttonStack.topAnchor, constant: -12),
            zoomFitButton.widthAnchor.constraint(equalToConstant: 44),

            buttonStack.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 12),
            buttonStack.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -12),
            buttonStack.bottomAnchor.constraint(equalTo: view.bottomAnchor, constant: -14),
            buttonStack.heightAnchor.constraint(equalToConstant: 28)
        ])

        setZoom(1)
    }

    func reload(entries: [ROIListEntry], selectedID: ROIIdentifier) {
        isProgrammaticSelection = true
        defer {
            isProgrammaticSelection = false
        }

        self.entries = entries
        self.selectedID = selectedID
        statusLabel.stringValue = "\(entries.filter(\.isComplete).count) of \(entries.count) complete"
        tableView.reloadData()

        if let selectedIndex = entries.firstIndex(where: { $0.id == selectedID }) {
            tableView.selectRowIndexes(IndexSet(integer: selectedIndex), byExtendingSelection: false)
            tableView.scrollRowToVisible(selectedIndex)
        }
    }

    func setZoom(_ zoom: CGFloat) {
        isUpdatingZoomControl = true
        defer {
            isUpdatingZoomControl = false
        }

        let zoomValue = Double(zoom)
        zoomValueLabel.stringValue = String(format: "%.2fx", zoomValue)
        zoomSlider.doubleValue = Self.sliderValue(for: zoomValue)
    }

    func numberOfRows(in tableView: NSTableView) -> Int {
        entries.count
    }

    func tableView(
        _ tableView: NSTableView,
        viewFor tableColumn: NSTableColumn?,
        row: Int
    ) -> NSView? {
        let identifier = NSUserInterfaceItemIdentifier("roiCell")
        let cell = tableView.makeView(withIdentifier: identifier, owner: self) as? NSTableCellView
            ?? makeCell(identifier: identifier)
        let entry = entries[row]
        let prefix = entry.isComplete ? "✓" : " "
        cell.textField?.stringValue = "\(prefix)  \(entry.label)"
        cell.textField?.textColor = entry.isComplete ? .labelColor : .secondaryLabelColor
        return cell
    }

    func tableViewSelectionDidChange(_ notification: Notification) {
        guard !isProgrammaticSelection else {
            return
        }
        let row = tableView.selectedRow
        guard entries.indices.contains(row) else {
            return
        }
        onSelectionChanged?(entries[row].id)
    }

    private func makeCell(identifier: NSUserInterfaceItemIdentifier) -> NSTableCellView {
        let cell = NSTableCellView()
        cell.identifier = identifier
        let textField = NSTextField(labelWithString: "")
        textField.lineBreakMode = .byTruncatingTail
        textField.translatesAutoresizingMaskIntoConstraints = false
        cell.addSubview(textField)
        cell.textField = textField
        NSLayoutConstraint.activate([
            textField.leadingAnchor.constraint(equalTo: cell.leadingAnchor, constant: 8),
            textField.trailingAnchor.constraint(equalTo: cell.trailingAnchor, constant: -8),
            textField.centerYAnchor.constraint(equalTo: cell.centerYAnchor)
        ])
        return cell
    }

    @objc private func nextPressed(_ sender: Any?) {
        onNext?()
    }

    @objc private func previousPressed(_ sender: Any?) {
        onPrevious?()
    }

    @objc private func clearPressed(_ sender: Any?) {
        onClear?()
    }

    @objc private func zoomSliderChanged(_ sender: NSSlider) {
        guard !isUpdatingZoomControl else {
            return
        }
        onZoomChanged?(CGFloat(Self.zoomValue(forSliderValue: sender.doubleValue)))
    }

    @objc private func zoomFitPressed(_ sender: Any?) {
        onResetZoom?()
    }

    private static func sliderValue(for zoom: Double) -> Double {
        let clampedZoom = min(max(zoom, sliderMinZoom), sliderMaxZoom)
        let minLog = log(sliderMinZoom)
        let maxLog = log(sliderMaxZoom)
        return (log(clampedZoom) - minLog) / (maxLog - minLog)
    }

    private static func zoomValue(forSliderValue value: Double) -> Double {
        let clampedValue = min(max(value, 0), 1)
        let minLog = log(sliderMinZoom)
        let maxLog = log(sliderMaxZoom)
        return exp(minLog + (maxLog - minLog) * clampedValue)
    }
}
