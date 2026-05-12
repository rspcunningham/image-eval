import Foundation

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

    enum CodingKeys: String, CodingKey, CaseIterable {
        case x0
        case y0
        case x1
        case y1
    }

    public init(from decoder: Decoder) throws {
        try rejectUnknownKeys(in: decoder, allowedKeys: CodingKeys.allCases, typeName: "PixelRect")
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.x0 = try container.decode(Int.self, forKey: .x0)
        self.y0 = try container.decode(Int.self, forKey: .y0)
        self.x1 = try container.decode(Int.self, forKey: .x1)
        self.y1 = try container.decode(Int.self, forKey: .y1)
    }
}

public struct SourceImage: Codable, Equatable, Sendable {
    public var width: Int
    public var height: Int

    public init(width: Int, height: Int) {
        self.width = width
        self.height = height
    }

    enum CodingKeys: String, CodingKey, CaseIterable {
        case width
        case height
    }

    public init(from decoder: Decoder) throws {
        try rejectUnknownKeys(in: decoder, allowedKeys: CodingKeys.allCases, typeName: "SourceImage")
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.width = try container.decode(Int.self, forKey: .width)
        self.height = try container.decode(Int.self, forKey: .height)
    }
}

public struct NormalizationROIs: Codable, Equatable, Sendable {
    public var black: PixelRect?
    public var white: PixelRect?

    public init(black: PixelRect? = nil, white: PixelRect? = nil) {
        self.black = black
        self.white = white
    }

    enum CodingKeys: String, CodingKey, CaseIterable {
        case black
        case white
    }

    public init(from decoder: Decoder) throws {
        try rejectUnknownKeys(in: decoder, allowedKeys: CodingKeys.allCases, typeName: "NormalizationROIs")
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try requireKey(.black, in: container)
        try requireKey(.white, in: container)
        self.black = try container.decodeIfPresent(PixelRect.self, forKey: .black)
        self.white = try container.decodeIfPresent(PixelRect.self, forKey: .white)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(black, forKey: .black)
        try container.encode(white, forKey: .white)
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

    enum CodingKeys: String, CodingKey, CaseIterable {
        case group
        case element
        case orientation
        case rect
    }

    public init(from decoder: Decoder) throws {
        try rejectUnknownKeys(in: decoder, allowedKeys: CodingKeys.allCases, typeName: "BarROI")
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try requireKey(.rect, in: container)
        self.group = try container.decode(Int.self, forKey: .group)
        self.element = try container.decode(Int.self, forKey: .element)
        self.orientation = try container.decode(String.self, forKey: .orientation)
        self.rect = try container.decodeIfPresent(PixelRect.self, forKey: .rect)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(group, forKey: .group)
        try container.encode(element, forKey: .element)
        try container.encode(orientation, forKey: .orientation)
        try container.encode(rect, forKey: .rect)
    }
}

public struct Template: Codable, Equatable, Sendable {
    public var sourceImage: SourceImage
    public var normalizationROIs: NormalizationROIs
    public var barROIs: [BarROI]

    public init(
        sourceImage: SourceImage,
        normalizationROIs: NormalizationROIs = NormalizationROIs(),
        barROIs: [BarROI]
    ) {
        self.sourceImage = sourceImage
        self.normalizationROIs = normalizationROIs
        self.barROIs = barROIs
    }

    enum CodingKeys: String, CodingKey, CaseIterable {
        case sourceImage = "source_image"
        case normalizationROIs = "normalization_rois"
        case barROIs = "bar_rois"
    }

    public init(from decoder: Decoder) throws {
        try rejectUnknownKeys(in: decoder, allowedKeys: CodingKeys.allCases, typeName: "Template")
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.sourceImage = try container.decode(SourceImage.self, forKey: .sourceImage)
        self.normalizationROIs = try container.decode(
            NormalizationROIs.self,
            forKey: .normalizationROIs
        )
        self.barROIs = try container.decode([BarROI].self, forKey: .barROIs)
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

private struct DynamicCodingKey: CodingKey {
    let stringValue: String
    let intValue: Int?

    init?(stringValue: String) {
        self.stringValue = stringValue
        self.intValue = nil
    }

    init?(intValue: Int) {
        self.stringValue = String(intValue)
        self.intValue = intValue
    }
}

private func rejectUnknownKeys<K: CodingKey>(
    in decoder: Decoder,
    allowedKeys: [K],
    typeName: String
) throws {
    let container = try decoder.container(keyedBy: DynamicCodingKey.self)
    let allowedKeyNames = Set(allowedKeys.map(\.stringValue))
    let unknownKeyNames = container.allKeys
        .map(\.stringValue)
        .filter { !allowedKeyNames.contains($0) }
        .sorted()

    guard let unknownKeyName = unknownKeyNames.first else {
        return
    }
    let unknownKey = DynamicCodingKey(stringValue: unknownKeyName)!
    throw DecodingError.dataCorrupted(DecodingError.Context(
        codingPath: decoder.codingPath + [unknownKey],
        debugDescription: "\(typeName) contains unknown key '\(unknownKeyName)'."
    ))
}

private func requireKey<K: CodingKey>(
    _ key: K,
    in container: KeyedDecodingContainer<K>
) throws {
    guard container.contains(key) else {
        throw DecodingError.keyNotFound(key, DecodingError.Context(
            codingPath: container.codingPath,
            debugDescription: "Missing required key '\(key.stringValue)'."
        ))
    }
}
