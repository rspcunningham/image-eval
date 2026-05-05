// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "ROISelector",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .library(name: "NPYCore", targets: ["NPYCore"]),
        .library(name: "ROISelectorCore", targets: ["ROISelectorCore"]),
        .executable(name: "ROISelector", targets: ["ROISelector"])
    ],
    targets: [
        .target(
            name: "NPYCore",
            path: "Sources/NPYCore"
        ),
        .target(
            name: "ROISelectorCore",
            dependencies: ["NPYCore"],
            path: "Sources/ROISelectorCore"
        ),
        .target(
            name: "ROISelectorApp",
            dependencies: ["NPYCore", "ROISelectorCore"],
            path: "Sources/ROISelectorApp"
        ),
        .executableTarget(
            name: "ROISelector",
            dependencies: ["ROISelectorApp"],
            path: "Sources/ROISelectorExecutable"
        ),
        .testTarget(
            name: "NPYCoreTests",
            dependencies: ["NPYCore"],
            path: "Tests/NPYCoreTests"
        ),
        .testTarget(
            name: "ROISelectorCoreTests",
            dependencies: ["NPYCore", "ROISelectorCore"],
            path: "Tests/ROISelectorCoreTests"
        )
    ],
    swiftLanguageModes: [.v5]
)
