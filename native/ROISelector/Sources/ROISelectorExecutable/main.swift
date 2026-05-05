import Foundation
import ROISelectorApp
import ROISelectorCore

do {
    let configuration = try AppConfiguration.parse(arguments: CommandLine.arguments)
    runROISelectorApplication(configuration: configuration)
} catch AppConfigurationError.helpRequested {
    print(AppConfiguration.usage)
    exit(0)
} catch {
    fputs("ROISelector: error: \(error.localizedDescription)\n", stderr)
    exit(1)
}
