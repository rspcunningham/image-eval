import Foundation

public enum TemplateDocumentError: LocalizedError, Equatable {
    case dimensionsMismatch(templateWidth: Int, templateHeight: Int, imageWidth: Int, imageHeight: Int)
    case missingGroupsForNewTemplate
    case missingElementsForNewTemplate
    case partialBarOverride
    case invalidBarOrientation(String)

    public var errorDescription: String? {
        switch self {
        case .dimensionsMismatch(let templateWidth, let templateHeight, let imageWidth, let imageHeight):
            "Template was initialized for \(templateWidth)x\(templateHeight), but the source image is \(imageWidth)x\(imageHeight)."
        case .missingGroupsForNewTemplate:
            "--groups is required when creating a new template."
        case .missingElementsForNewTemplate:
            "--elements is required when creating a new template."
        case .partialBarOverride:
            "Pass both --groups and --elements to replace an existing template's bar ROI list."
        case .invalidBarOrientation(let orientation):
            "Invalid bar ROI orientation \(orientation). Expected X or Y."
        }
    }
}

public final class TemplateDocument {
    public let url: URL
    public private(set) var template: Template

    public init(url: URL, template: Template) {
        self.url = url
        self.template = template
    }

    public static func loadOrCreate(
        sourceURL: URL,
        templateURL: URL,
        imageWidth: Int,
        imageHeight: Int,
        groupsSpec: String?,
        elementsSpec: String?
    ) throws -> TemplateDocument {
        if FileManager.default.fileExists(atPath: templateURL.path) {
            var template = try decodeTemplate(from: templateURL)
            try validate(template: template, imageWidth: imageWidth, imageHeight: imageHeight)

            if groupsSpec != nil || elementsSpec != nil {
                guard let groupsSpec, let elementsSpec else {
                    throw TemplateDocumentError.partialBarOverride
                }
                let groups = try IntegerRangeParser.parse(groupsSpec, label: "groups")
                let elements = try IntegerRangeParser.parse(
                    elementsSpec,
                    label: "elements",
                    minimum: 1
                )
                template.barROIs = makeBarROIs(
                    groups: groups,
                    elements: elements,
                    preserving: template.barROIs
                )
            }

            template.sourceImage = SourceImage(
                path: sourceURL.path,
                width: imageWidth,
                height: imageHeight
            )
            template.baseImagePath = sourceURL.path
            try validateBarROIs(template.barROIs)
            let document = TemplateDocument(url: templateURL, template: template)
            try document.save()
            return document
        }

        guard let groupsSpec else {
            throw TemplateDocumentError.missingGroupsForNewTemplate
        }
        guard let elementsSpec else {
            throw TemplateDocumentError.missingElementsForNewTemplate
        }

        let groups = try IntegerRangeParser.parse(groupsSpec, label: "groups")
        let elements = try IntegerRangeParser.parse(elementsSpec, label: "elements", minimum: 1)
        let template = Template(
            sourceImage: SourceImage(path: sourceURL.path, width: imageWidth, height: imageHeight),
            baseImagePath: sourceURL.path,
            barROIs: makeBarROIs(groups: groups, elements: elements)
        )
        let document = TemplateDocument(url: templateURL, template: template)
        try document.save()
        return document
    }

    public func save() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(template)
        let directory = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        try data.write(to: url, options: [.atomic])
    }

    public func entries() -> [ROIListEntry] {
        var entries: [ROIListEntry] = [
            ROIListEntry(
                id: .normalization(.black),
                label: NormalizationROIName.black.label,
                rect: template.normalizationROIs.black
            ),
            ROIListEntry(
                id: .normalization(.white),
                label: NormalizationROIName.white.label,
                rect: template.normalizationROIs.white
            )
        ]

        for index in template.barROIs.indices {
            let roi = template.barROIs[index]
            entries.append(ROIListEntry(
                id: .bar(index),
                label: "G\(roi.group) E\(roi.element) \(roi.orientation)",
                rect: roi.rect
            ))
        }
        return entries
    }

    public func rect(for id: ROIIdentifier) -> PixelRect? {
        switch id {
        case .normalization(.black):
            template.normalizationROIs.black
        case .normalization(.white):
            template.normalizationROIs.white
        case .bar(let index):
            template.barROIs.indices.contains(index) ? template.barROIs[index].rect : nil
        }
    }

    public func setRect(_ rect: PixelRect?, for id: ROIIdentifier) throws {
        let clamped = rect?.clamped(
            width: template.sourceImage.width,
            height: template.sourceImage.height
        )
        switch id {
        case .normalization(.black):
            template.normalizationROIs.black = clamped
        case .normalization(.white):
            template.normalizationROIs.white = clamped
        case .bar(let index):
            guard template.barROIs.indices.contains(index) else {
                return
            }
            template.barROIs[index].rect = clamped
        }
        try save()
    }
}

private func decodeTemplate(from url: URL) throws -> Template {
    let data = try Data(contentsOf: url)
    let decoder = JSONDecoder()
    let template = try decoder.decode(Template.self, from: data)
    return template
}

private func validate(template: Template, imageWidth: Int, imageHeight: Int) throws {
    guard template.sourceImage.width == imageWidth, template.sourceImage.height == imageHeight else {
        throw TemplateDocumentError.dimensionsMismatch(
            templateWidth: template.sourceImage.width,
            templateHeight: template.sourceImage.height,
            imageWidth: imageWidth,
            imageHeight: imageHeight
        )
    }
    try validateBarROIs(template.barROIs)
}

private func makeBarROIs(
    groups: [Int],
    elements: [Int],
    preserving existing: [BarROI] = []
) -> [BarROI] {
    let existingByKey = existing.reduce(into: [String: BarROI]()) { entries, roi in
        entries[roi.identityKey] = roi
    }
    var rois: [BarROI] = []
    for group in groups {
        for element in elements {
            for orientation in fixedOrientations {
                let key = "\(group):\(element):\(orientation)"
                rois.append(existingByKey[key] ?? BarROI(
                    group: group,
                    element: element,
                    orientation: orientation
                ))
            }
        }
    }
    return rois
}

private func validateBarROIs(_ rois: [BarROI]) throws {
    for roi in rois {
        guard fixedOrientations.contains(roi.orientation) else {
            throw TemplateDocumentError.invalidBarOrientation(roi.orientation)
        }
    }
}
