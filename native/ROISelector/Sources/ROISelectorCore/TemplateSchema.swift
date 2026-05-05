import Foundation

public let templateSchemaVersion = 2
public let fixedOrientations = ["X", "Y"]

public struct PixelRect: Codable, Equatable, Sendable {
    public var x0: Int
    public var y0: Int
    public var x1: Int
    public var y1: Int

    public init(x0: Int, y0: Int, x1: Int, y1: Int) {
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
    }

    public var width: Int { x1 - x0 }
    public var height: Int { y1 - y0 }
    public var isValid: Bool { width > 0 && height > 0 }

    public func clamped(width imageWidth: Int, height imageHeight: Int) -> PixelRect? {
        let clampedX0 = min(max(x0, 0), imageWidth)
        let clampedY0 = min(max(y0, 0), imageHeight)
        let clampedX1 = min(max(x1, 0), imageWidth)
        let clampedY1 = min(max(y1, 0), imageHeight)
        let normalized = PixelRect(
            x0: min(clampedX0, clampedX1),
            y0: min(clampedY0, clampedY1),
            x1: max(clampedX0, clampedX1),
            y1: max(clampedY0, clampedY1)
        )
        return normalized.isValid ? normalized : nil
    }
}

public struct SourceImage: Codable, Equatable, Sendable {
    public var path: String
    public var width: Int
    public var height: Int

    public init(path: String, width: Int, height: Int) {
        self.path = path
        self.width = width
        self.height = height
    }
}

public struct NormalizationROIs: Codable, Equatable, Sendable {
    public var black: PixelRect?
    public var white: PixelRect?

    public init(black: PixelRect? = nil, white: PixelRect? = nil) {
        self.black = black
        self.white = white
    }
}

public struct BarROI: Codable, Equatable, Sendable {
    public var group: Int
    public var element: Int
    public var orientation: String
    public var rect: PixelRect?

    public init(group: Int, element: Int, orientation: String, rect: PixelRect? = nil) {
        self.group = group
        self.element = element
        self.orientation = orientation
        self.rect = rect
    }

    public var identityKey: String {
        "\(group):\(element):\(orientation)"
    }
}

public struct Template: Codable, Equatable, Sendable {
    public var schemaVersion: Int
    public var sourceImage: SourceImage
    public var anchor: PixelRect?
    public var normalizationROIs: NormalizationROIs
    public var barROIs: [BarROI]

    public init(
        schemaVersion: Int = templateSchemaVersion,
        sourceImage: SourceImage,
        anchor: PixelRect? = nil,
        normalizationROIs: NormalizationROIs = NormalizationROIs(),
        barROIs: [BarROI]
    ) {
        self.schemaVersion = schemaVersion
        self.sourceImage = sourceImage
        self.anchor = anchor
        self.normalizationROIs = normalizationROIs
        self.barROIs = barROIs
    }

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case sourceImage = "source_image"
        case anchor
        case normalizationROIs = "normalization_rois"
        case barROIs = "bar_rois"
    }
}

public enum NormalizationROIName: String, CaseIterable, Sendable {
    case black
    case white

    public var label: String {
        switch self {
        case .black:
            "Black normalization"
        case .white:
            "White normalization"
        }
    }
}

public enum ROIIdentifier: Hashable, Sendable {
    case anchor
    case normalization(NormalizationROIName)
    case bar(Int)
}

public struct ROIListEntry: Equatable, Sendable {
    public var id: ROIIdentifier
    public var label: String
    public var rect: PixelRect?

    public init(id: ROIIdentifier, label: String, rect: PixelRect?) {
        self.id = id
        self.label = label
        self.rect = rect
    }

    public var isComplete: Bool {
        rect?.isValid == true
    }
}
