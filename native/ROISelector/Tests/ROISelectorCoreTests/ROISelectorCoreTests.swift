import Foundation
import Testing
import NPYCore
@testable import ROISelectorCore

@Test func parsesRangeSpecs() throws {
    #expect(try IntegerRangeParser.parse("4-7", label: "groups") == [4, 5, 6, 7])
    #expect(try IntegerRangeParser.parse("7-4", label: "groups") == [7, 6, 5, 4])
    #expect(try IntegerRangeParser.parse("-1..1", label: "groups") == [-1, 0, 1])
    #expect(try IntegerRangeParser.parse("1,3", label: "elements", minimum: 1) == [1, 3])
}

@Test func parsesCLIArguments() throws {
    let config = try AppConfiguration.parse(arguments: [
        "ROISelector",
        "source.npy",
        "template.json",
        "--groups=4-5",
        "--elements",
        "1,2"
    ])

    #expect(config.sourceURL.path.hasSuffix("source.npy"))
    #expect(config.templateURL.path.hasSuffix("template.json"))
    #expect(config.groupsSpec == "4-5")
    #expect(config.elementsSpec == "1,2")
}

@Test func templateJSONRoundTripsWithSnakeCase() throws {
    let template = Template(
        sourceImage: SourceImage(path: "sample.npy", width: 20, height: 10),
        anchor: PixelRect(x0: 1, y0: 2, x1: 3, y1: 4),
        normalizationROIs: NormalizationROIs(
            black: PixelRect(x0: 0, y0: 0, x1: 2, y1: 2),
            white: nil
        ),
        barROIs: [
            BarROI(
                group: 4,
                element: 1,
                orientation: "X",
                rect: PixelRect(x0: 5, y0: 6, x1: 8, y1: 9)
            )
        ]
    )

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(template)
    let text = String(decoding: data, as: UTF8.self)
    #expect(text.contains("schema_version"))
    #expect(text.contains("source_image"))
    #expect(text.contains("normalization_rois"))
    #expect(text.contains("bar_rois"))

    let decoded = try JSONDecoder().decode(Template.self, from: data)
    #expect(decoded == template)
}

@Test func createsNewTemplateDocument() throws {
    let directory = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let templateURL = directory.appendingPathComponent("template.json")

    let document = try TemplateDocument.loadOrCreate(
        sourceURL: URL(fileURLWithPath: "/tmp/source.npy"),
        templateURL: templateURL,
        imageWidth: 100,
        imageHeight: 80,
        groupsSpec: "4-5",
        elementsSpec: "1,2"
    )

    #expect(document.template.schemaVersion == 2)
    #expect(document.template.barROIs.count == 8)
    #expect(document.template.barROIs.map(\.orientation) == ["X", "Y", "X", "Y", "X", "Y", "X", "Y"])
    #expect(FileManager.default.fileExists(atPath: templateURL.path))
}

@Test func reusesExistingTemplateBars() throws {
    let directory = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let templateURL = directory.appendingPathComponent("template.json")

    let original = Template(
        sourceImage: SourceImage(path: "/tmp/source.npy", width: 20, height: 10),
        barROIs: [
            BarROI(group: 9, element: 2, orientation: "Y", rect: PixelRect(x0: 1, y0: 1, x1: 3, y1: 4))
        ]
    )
    let data = try JSONEncoder().encode(original)
    try data.write(to: templateURL)

    let document = try TemplateDocument.loadOrCreate(
        sourceURL: URL(fileURLWithPath: "/tmp/source.npy"),
        templateURL: templateURL,
        imageWidth: 20,
        imageHeight: 10,
        groupsSpec: nil,
        elementsSpec: nil
    )

    #expect(document.template.barROIs == original.barROIs)
}

@Test func displayImageConvertsFloat32() throws {
    let array = try NPYArray(data: makeNPY(
        descr: "<f4",
        shape: [2, 2],
        payload: floats([0, 1, 2, 3])
    ))

    let display = try NPYDisplayImage(array: array)
    #expect(display.width == 2)
    #expect(display.height == 2)
    #expect(display.cgImage.width == 2)
    #expect(display.cgImage.height == 2)
}

private func temporaryDirectory() throws -> URL {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("ROISelectorCoreTests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    return directory
}

private func makeNPY(
    descr: String,
    shape: [Int],
    payload: Data,
    major: UInt8 = 1,
    minor: UInt8 = 0
) -> Data {
    let shapeText = shape.map(String.init).joined(separator: ", ") + (shape.count == 1 ? "," : "")
    let header = "{'descr': '\(descr)', 'fortran_order': False, 'shape': (\(shapeText)), }"
    var data = Data([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, major, minor])
    var paddedHeader = header
    let lengthByteCount = major == 1 ? 2 : 4
    let baseLength = 6 + 2 + lengthByteCount
    let paddedHeaderLength =
        ((baseLength + paddedHeader.utf8.count + 1 + 15) / 16) * 16 - baseLength
    paddedHeader += String(
        repeating: " ",
        count: paddedHeaderLength - paddedHeader.utf8.count - 1
    )
    paddedHeader += "\n"

    let length = UInt32(paddedHeader.utf8.count)
    data.append(UInt8(length & 0xff))
    data.append(UInt8((length >> 8) & 0xff))
    if lengthByteCount == 4 {
        data.append(UInt8((length >> 16) & 0xff))
        data.append(UInt8((length >> 24) & 0xff))
    }
    data.append(paddedHeader.data(using: major == 3 ? .utf8 : .isoLatin1)!)
    data.append(payload)
    return data
}

private func floats(_ values: [Float]) -> Data {
    var values = values
    return Data(bytes: &values, count: values.count * MemoryLayout<Float>.size)
}
