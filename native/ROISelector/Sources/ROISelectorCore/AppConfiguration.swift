import Foundation

public struct AppConfiguration: Equatable, Sendable {
    public var sourceURL: URL
    public var templateURL: URL
    public var groupsSpec: String?
    public var elementsSpec: String?

    public init(
        sourceURL: URL,
        templateURL: URL,
        groupsSpec: String? = nil,
        elementsSpec: String? = nil
    ) {
        self.sourceURL = sourceURL
        self.templateURL = templateURL
        self.groupsSpec = groupsSpec
        self.elementsSpec = elementsSpec
    }
}

public enum AppConfigurationError: LocalizedError, Equatable {
    case helpRequested
    case missingValue(String)
    case unknownOption(String)
    case wrongArgumentCount(Int)

    public var errorDescription: String? {
        switch self {
        case .helpRequested:
            AppConfiguration.usage
        case .missingValue(let option):
            "\(option) requires a value."
        case .unknownOption(let option):
            "Unknown option: \(option)."
        case .wrongArgumentCount(let count):
            "Expected source image path and template path, got \(count) positional arguments."
        }
    }
}

public extension AppConfiguration {
    static let usage = """
    Usage: ROISelector source.npy template.json [--groups SPEC] [--elements SPEC]

    Options:
      --groups SPEC      Group selection, such as 4-7 or 4,5,7. Required for new templates.
      --elements SPEC    Element selection, such as 1-6 or 1,3,6. Required for new templates.
      -h, --help         Show this help.
    """

    static func parse(arguments: [String]) throws -> AppConfiguration {
        var groupsSpec: String?
        var elementsSpec: String?
        var positional: [String] = []

        var index = 1
        while index < arguments.count {
            let argument = arguments[index]
            if argument == "-h" || argument == "--help" {
                throw AppConfigurationError.helpRequested
            } else if argument == "--groups" {
                index += 1
                guard index < arguments.count else {
                    throw AppConfigurationError.missingValue("--groups")
                }
                groupsSpec = arguments[index]
            } else if argument.hasPrefix("--groups=") {
                groupsSpec = String(argument.dropFirst("--groups=".count))
            } else if argument == "--elements" {
                index += 1
                guard index < arguments.count else {
                    throw AppConfigurationError.missingValue("--elements")
                }
                elementsSpec = arguments[index]
            } else if argument.hasPrefix("--elements=") {
                elementsSpec = String(argument.dropFirst("--elements=".count))
            } else if argument.hasPrefix("-") {
                throw AppConfigurationError.unknownOption(argument)
            } else {
                positional.append(argument)
            }
            index += 1
        }

        guard positional.count == 2 else {
            throw AppConfigurationError.wrongArgumentCount(positional.count)
        }

        return AppConfiguration(
            sourceURL: URL(fileURLWithPath: positional[0]),
            templateURL: URL(fileURLWithPath: positional[1]),
            groupsSpec: groupsSpec,
            elementsSpec: elementsSpec
        )
    }
}
