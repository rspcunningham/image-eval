import Foundation

public enum IntegerRangeParserError: LocalizedError, Equatable {
    case emptySelection(String)
    case invalidSelection(label: String, value: String)
    case duplicateSelection(label: String, value: Int)
    case belowMinimum(label: String, value: Int, minimum: Int)

    public var errorDescription: String? {
        switch self {
        case .emptySelection(let label):
            "\(label) contains an empty selection."
        case .invalidSelection(let label, let value):
            "Invalid \(label) selection: \(value)."
        case .duplicateSelection(let label, let value):
            "\(label) contains duplicate value: \(value)."
        case .belowMinimum(let label, let value, let minimum):
            "\(label) value \(value) must be >= \(minimum)."
        }
    }
}

public struct IntegerRangeParser {
    private static let rangeRegex = try! NSRegularExpression(
        pattern: #"^([+-]?\d+)\s*(?:-|\.\.)\s*([+-]?\d+)$"#
    )

    public static func parse(
        _ spec: String,
        label: String,
        minimum: Int? = nil
    ) throws -> [Int] {
        var values: [Int] = []
        for rawPart in spec.split(separator: ",", omittingEmptySubsequences: false) {
            let part = rawPart.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !part.isEmpty else {
                throw IntegerRangeParserError.emptySelection(label)
            }

            if let range = try parseRange(part) {
                let step = range.start <= range.end ? 1 : -1
                values.append(contentsOf: stride(
                    from: range.start,
                    through: range.end,
                    by: step
                ))
            } else if let value = Int(part) {
                values.append(value)
            } else {
                throw IntegerRangeParserError.invalidSelection(label: label, value: part)
            }
        }

        guard !values.isEmpty else {
            throw IntegerRangeParserError.emptySelection(label)
        }

        var seen: Set<Int> = []
        for value in values {
            if let minimum, value < minimum {
                throw IntegerRangeParserError.belowMinimum(
                    label: label,
                    value: value,
                    minimum: minimum
                )
            }
            guard seen.insert(value).inserted else {
                throw IntegerRangeParserError.duplicateSelection(label: label, value: value)
            }
        }
        return values
    }

    private static func parseRange(_ text: String) throws -> (start: Int, end: Int)? {
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        guard
            let match = rangeRegex.firstMatch(in: text, range: range),
            match.numberOfRanges == 3,
            let startRange = Range(match.range(at: 1), in: text),
            let endRange = Range(match.range(at: 2), in: text),
            let start = Int(text[startRange]),
            let end = Int(text[endRange])
        else {
            return nil
        }
        return (start, end)
    }
}
