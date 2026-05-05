import AppKit
import ROISelectorCore

final class ROICanvasView: NSView {
    static let minimumZoom: CGFloat = 0.03
    static let maximumZoom: CGFloat = 80

    private enum DragMode {
        case draw(start: CGPoint)
        case move(start: CGPoint, rect: PixelRect)
        case resize(handle: ResizeHandle)
        case pan(start: CGPoint, pan: CGPoint)
    }

    private enum ResizeHandle {
        case topLeft
        case topRight
        case bottomRight
        case bottomLeft
        case left
        case right
        case top
        case bottom
    }

    private var displayImage: NPYDisplayImage?
    private var image: NSImage?
    private var entries: [ROIListEntry] = []
    private var activeID: ROIIdentifier?
    private var zoom: CGFloat = 1
    private var pan: CGPoint = .zero
    private var didFitInitialView = false
    private var dragMode: DragMode?
    private var dragStartImagePoint: CGPoint?
    private var spaceIsDown = false

    var onRectChanged: ((PixelRect?) -> Void)?
    var onAdvanceRequested: (() -> Void)?
    var onClearRequested: (() -> Void)?
    var onZoomChanged: ((CGFloat) -> Void)?

    var currentZoom: CGFloat {
        zoom
    }

    override var isFlipped: Bool {
        true
    }

    override var acceptsFirstResponder: Bool {
        true
    }

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        configureLayer()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        configureLayer()
    }

    func configure(
        displayImage: NPYDisplayImage,
        entries: [ROIListEntry],
        activeID: ROIIdentifier
    ) {
        if self.displayImage?.width != displayImage.width || self.displayImage?.height != displayImage.height {
            didFitInitialView = false
        }
        self.displayImage = displayImage
        self.image = NSImage(
            cgImage: displayImage.cgImage,
            size: NSSize(width: displayImage.width, height: displayImage.height)
        )
        self.entries = entries
        self.activeID = activeID
        if !didFitInitialView, bounds.width > 0, bounds.height > 0 {
            resetView()
            didFitInitialView = true
        }
        needsDisplay = true
    }

    func resetView() {
        guard let displayImage else {
            return
        }
        let availableWidth = max(bounds.width - 32, 1)
        let availableHeight = max(bounds.height - 32, 1)
        zoom = min(
            availableWidth / CGFloat(displayImage.width),
            availableHeight / CGFloat(displayImage.height)
        )
        pan = CGPoint(
            x: (bounds.width - CGFloat(displayImage.width) * zoom) / 2,
            y: (bounds.height - CGFloat(displayImage.height) * zoom) / 2
        )
        needsDisplay = true
        notifyZoomChanged()
    }

    func setZoom(_ newZoom: CGFloat) {
        zoom(to: newZoom, around: CGPoint(x: bounds.midX, y: bounds.midY))
    }

    override func layout() {
        super.layout()
        if !didFitInitialView, displayImage != nil {
            resetView()
            didFitInitialView = true
        }
    }

    override func draw(_ dirtyRect: NSRect) {
        NSGraphicsContext.saveGraphicsState()
        NSBezierPath(rect: bounds).addClip()
        defer {
            NSGraphicsContext.restoreGraphicsState()
        }

        NSColor.black.setFill()
        bounds.fill()

        guard let image, let displayImage else {
            drawEmptyState()
            return
        }

        image.draw(
            in: imageRect(for: displayImage),
            from: NSRect(x: 0, y: 0, width: displayImage.width, height: displayImage.height),
            operation: .copy,
            fraction: 1,
            respectFlipped: true,
            hints: [.interpolation: NSImageInterpolation.none]
        )

        for entry in entries where entry.id != activeID {
            if let rect = entry.rect {
                draw(rect: rect, label: entry.label, color: NSColor.systemGray, alpha: 0.45, active: false)
            }
        }

        if let activeEntry, let rect = activeEntry.rect {
            draw(rect: rect, label: activeEntry.label, color: NSColor.systemCyan, alpha: 1.0, active: true)
        }

        drawHUD()
    }

    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
        let point = convert(event.locationInWindow, from: nil)
        if spaceIsDown || event.modifierFlags.contains(.shift) {
            dragMode = .pan(start: point, pan: pan)
            return
        }

        guard let imagePoint = imagePoint(from: point) else {
            return
        }
        dragStartImagePoint = imagePoint

        if let rect = activeEntry?.rect {
            if let handle = hitResizeHandle(point: point, rect: rect) {
                dragMode = .resize(handle: handle)
                updateResize(handle: handle, imagePoint: imagePoint)
            } else if viewRect(for: rect).contains(point) {
                dragMode = .move(start: imagePoint, rect: rect)
            } else {
                dragMode = .draw(start: imagePoint)
                setActiveRect(rectFromPoints(imagePoint, imagePoint))
            }
        } else {
            dragMode = .draw(start: imagePoint)
            setActiveRect(rectFromPoints(imagePoint, imagePoint))
        }
    }

    override func mouseDragged(with event: NSEvent) {
        let point = convert(event.locationInWindow, from: nil)
        switch dragMode {
        case .draw(let start):
            guard let imagePoint = imagePoint(from: point) else {
                return
            }
            setActiveRect(rectFromPoints(start, imagePoint))
        case .move(let start, let rect):
            guard let imagePoint = imagePoint(from: point), let displayImage else {
                return
            }
            let dx = Int((imagePoint.x - start.x).rounded())
            let dy = Int((imagePoint.y - start.y).rounded())
            setActiveRect(PixelRect(
                x0: rect.x0 + dx,
                y0: rect.y0 + dy,
                x1: rect.x1 + dx,
                y1: rect.y1 + dy
            ).clamped(width: displayImage.width, height: displayImage.height))
        case .resize(let handle):
            guard let imagePoint = imagePoint(from: point) else {
                return
            }
            updateResize(handle: handle, imagePoint: imagePoint)
        case .pan(let start, let originalPan):
            pan = CGPoint(
                x: originalPan.x + point.x - start.x,
                y: originalPan.y + point.y - start.y
            )
            needsDisplay = true
        case nil:
            return
        }
    }

    override func mouseUp(with event: NSEvent) {
        dragMode = nil
        dragStartImagePoint = nil
    }

    override func rightMouseDown(with event: NSEvent) {
        let point = convert(event.locationInWindow, from: nil)
        dragMode = .pan(start: point, pan: pan)
    }

    override func rightMouseDragged(with event: NSEvent) {
        mouseDragged(with: event)
    }

    override func rightMouseUp(with event: NSEvent) {
        mouseUp(with: event)
    }

    override func otherMouseDown(with event: NSEvent) {
        let point = convert(event.locationInWindow, from: nil)
        dragMode = .pan(start: point, pan: pan)
    }

    override func otherMouseDragged(with event: NSEvent) {
        mouseDragged(with: event)
    }

    override func otherMouseUp(with event: NSEvent) {
        mouseUp(with: event)
    }

    override func scrollWheel(with event: NSEvent) {
        let point = convert(event.locationInWindow, from: nil)
        if event.modifierFlags.contains(.shift) {
            pan.x -= event.scrollingDeltaY
            needsDisplay = true
            return
        }

        let delta = event.scrollingDeltaY
        guard delta != 0 else {
            return
        }
        let magnitude = min(abs(delta), 12) * 0.006
        let factor = delta > 0 ? 1 + magnitude : 1 / (1 + magnitude)
        zoom(by: factor, around: point)
    }

    override func magnify(with event: NSEvent) {
        let point = convert(event.locationInWindow, from: nil)
        zoom(by: max(0.2, 1 + event.magnification), around: point)
    }

    override func keyDown(with event: NSEvent) {
        switch event.keyCode {
        case 49:
            spaceIsDown = true
        case 36, 76:
            onAdvanceRequested?()
        case 51, 117:
            onClearRequested?()
        case 24, 69:
            zoom(by: 1.2, around: CGPoint(x: bounds.midX, y: bounds.midY))
        case 27, 78:
            zoom(by: 1 / 1.2, around: CGPoint(x: bounds.midX, y: bounds.midY))
        case 29:
            resetView()
        default:
            super.keyDown(with: event)
        }
    }

    override func keyUp(with event: NSEvent) {
        if event.keyCode == 49 {
            spaceIsDown = false
        } else {
            super.keyUp(with: event)
        }
    }

    override func resetCursorRects() {
        super.resetCursorRects()
        addCursorRect(bounds, cursor: .crosshair)
    }

    private var activeEntry: ROIListEntry? {
        guard let activeID else {
            return nil
        }
        return entries.first { $0.id == activeID }
    }

    private func configureLayer() {
        wantsLayer = true
        layer?.backgroundColor = NSColor.black.cgColor
        layer?.masksToBounds = true
    }

    private func setActiveRect(_ rect: PixelRect?) {
        onRectChanged?(rect)
        needsDisplay = true
    }

    private func updateResize(handle: ResizeHandle, imagePoint: CGPoint) {
        guard var rect = activeEntry?.rect else {
            return
        }
        let x = Int(imagePoint.x.rounded())
        let y = Int(imagePoint.y.rounded())
        switch handle {
        case .topLeft:
            rect.x0 = x
            rect.y0 = y
        case .topRight:
            rect.x1 = x
            rect.y0 = y
        case .bottomRight:
            rect.x1 = x
            rect.y1 = y
        case .bottomLeft:
            rect.x0 = x
            rect.y1 = y
        case .left:
            rect.x0 = x
        case .right:
            rect.x1 = x
        case .top:
            rect.y0 = y
        case .bottom:
            rect.y1 = y
        }
        setActiveRect(normalized(rect))
    }

    private func normalized(_ rect: PixelRect) -> PixelRect? {
        guard let displayImage else {
            return nil
        }
        return PixelRect(
            x0: min(rect.x0, rect.x1),
            y0: min(rect.y0, rect.y1),
            x1: max(rect.x0, rect.x1),
            y1: max(rect.y0, rect.y1)
        ).clamped(width: displayImage.width, height: displayImage.height)
    }

    private func rectFromPoints(_ a: CGPoint, _ b: CGPoint) -> PixelRect? {
        guard let displayImage else {
            return nil
        }
        let x0 = Int(floor(min(a.x, b.x)))
        let y0 = Int(floor(min(a.y, b.y)))
        let x1 = Int(ceil(max(a.x, b.x)))
        let y1 = Int(ceil(max(a.y, b.y)))
        return PixelRect(
            x0: x0,
            y0: y0,
            x1: max(x1, x0 + 1),
            y1: max(y1, y0 + 1)
        ).clamped(width: displayImage.width, height: displayImage.height)
    }

    private func imageRect(for displayImage: NPYDisplayImage) -> NSRect {
        NSRect(
            x: pan.x,
            y: pan.y,
            width: CGFloat(displayImage.width) * zoom,
            height: CGFloat(displayImage.height) * zoom
        )
    }

    private func viewRect(for rect: PixelRect) -> NSRect {
        NSRect(
            x: pan.x + CGFloat(rect.x0) * zoom,
            y: pan.y + CGFloat(rect.y0) * zoom,
            width: CGFloat(rect.width) * zoom,
            height: CGFloat(rect.height) * zoom
        )
    }

    private func imagePoint(from point: CGPoint) -> CGPoint? {
        guard let displayImage else {
            return nil
        }
        let x = (point.x - pan.x) / zoom
        let y = (point.y - pan.y) / zoom
        guard x >= 0, y >= 0, x <= CGFloat(displayImage.width), y <= CGFloat(displayImage.height) else {
            return nil
        }
        return CGPoint(x: x, y: y)
    }

    private func zoom(by factor: CGFloat, around point: CGPoint) {
        zoom(to: zoom * factor, around: point)
    }

    private func zoom(to requestedZoom: CGFloat, around point: CGPoint) {
        let oldZoom = zoom
        let newZoom = min(max(requestedZoom, Self.minimumZoom), Self.maximumZoom)
        guard newZoom != oldZoom else {
            return
        }

        let imageX = (point.x - pan.x) / oldZoom
        let imageY = (point.y - pan.y) / oldZoom
        zoom = newZoom
        pan = CGPoint(
            x: point.x - imageX * newZoom,
            y: point.y - imageY * newZoom
        )
        needsDisplay = true
        notifyZoomChanged()
    }

    private func notifyZoomChanged() {
        onZoomChanged?(zoom)
    }

    private func hitResizeHandle(point: CGPoint, rect: PixelRect) -> ResizeHandle? {
        let viewRect = viewRect(for: rect)
        let tolerance: CGFloat = 8
        let handles: [(ResizeHandle, CGPoint)] = [
            (.topLeft, CGPoint(x: viewRect.minX, y: viewRect.minY)),
            (.topRight, CGPoint(x: viewRect.maxX, y: viewRect.minY)),
            (.bottomRight, CGPoint(x: viewRect.maxX, y: viewRect.maxY)),
            (.bottomLeft, CGPoint(x: viewRect.minX, y: viewRect.maxY))
        ]
        for (handle, handlePoint) in handles {
            if hypot(point.x - handlePoint.x, point.y - handlePoint.y) <= tolerance {
                return handle
            }
        }

        if abs(point.x - viewRect.minX) <= tolerance, viewRect.minY...viewRect.maxY ~= point.y {
            return .left
        }
        if abs(point.x - viewRect.maxX) <= tolerance, viewRect.minY...viewRect.maxY ~= point.y {
            return .right
        }
        if abs(point.y - viewRect.minY) <= tolerance, viewRect.minX...viewRect.maxX ~= point.x {
            return .top
        }
        if abs(point.y - viewRect.maxY) <= tolerance, viewRect.minX...viewRect.maxX ~= point.x {
            return .bottom
        }
        return nil
    }

    private func draw(rect: PixelRect, label: String, color: NSColor, alpha: CGFloat, active: Bool) {
        let viewRect = viewRect(for: rect)
        guard viewRect.intersects(bounds) else {
            return
        }

        let strokeColor = color.withAlphaComponent(alpha)
        strokeColor.setStroke()
        let path = NSBezierPath(rect: viewRect)
        path.lineWidth = active ? 2 : 1
        path.stroke()

        if active {
            strokeColor.setFill()
            for point in [
                CGPoint(x: viewRect.minX, y: viewRect.minY),
                CGPoint(x: viewRect.maxX, y: viewRect.minY),
                CGPoint(x: viewRect.maxX, y: viewRect.maxY),
                CGPoint(x: viewRect.minX, y: viewRect.maxY)
            ] {
                NSBezierPath(ovalIn: NSRect(x: point.x - 4, y: point.y - 4, width: 8, height: 8)).fill()
            }
        }

        let attributes: [NSAttributedString.Key: Any] = [
            .foregroundColor: strokeColor,
            .font: NSFont.systemFont(ofSize: active ? 12 : 10, weight: active ? .semibold : .regular)
        ]
        NSString(string: label).draw(
            at: CGPoint(x: max(viewRect.minX, 8), y: max(viewRect.minY - 18, 8)),
            withAttributes: attributes
        )
    }

    private func drawHUD() {
        let title = activeEntry?.label ?? "No ROI"
        let text = "\(title)   zoom \(String(format: "%.2f", Double(zoom)))x"
        let padding: CGFloat = 8
        let attributes: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 13, weight: .semibold),
            .foregroundColor: NSColor.white
        ]
        let size = NSString(string: text).size(withAttributes: attributes)
        let rect = NSRect(x: 14, y: 14, width: size.width + padding * 2, height: size.height + padding * 2)
        NSColor.black.withAlphaComponent(0.65).setFill()
        NSBezierPath(roundedRect: rect, xRadius: 5, yRadius: 5).fill()
        NSString(string: text).draw(
            at: CGPoint(x: rect.minX + padding, y: rect.minY + padding),
            withAttributes: attributes
        )
    }

    private func drawEmptyState() {
        let text = "No image loaded"
        let attributes: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 18, weight: .medium),
            .foregroundColor: NSColor.secondaryLabelColor
        ]
        let size = NSString(string: text).size(withAttributes: attributes)
        NSString(string: text).draw(
            at: CGPoint(x: bounds.midX - size.width / 2, y: bounds.midY - size.height / 2),
            withAttributes: attributes
        )
    }
}
