use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set"));
    let profile = env::var("PROFILE").unwrap_or_else(|_| String::from("debug"));
    let config = if profile == "release" {
        "Release"
    } else {
        "Debug"
    };

    println!("cargo:rerun-if-env-changed=AIONE_DIR");
    println!("cargo:rerun-if-env-changed=AIONE_LIB_DIR");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_TTS");

    let enable_tts = env::var_os("CARGO_FEATURE_TTS").is_some();

    if let Some(prefix) = env::var_os("AIONE_LIB_DIR").map(PathBuf::from) {
        link_prebuilt(&prefix, &out_dir)
            .unwrap_or_else(|error| panic!("Failed to stage AIOne runtime libraries: {error}"));
        return;
    }

    let aione_dir = find_aione_dir(&manifest_dir).expect(
        "Could not find the AIOne sources. Set AIONE_DIR or point AIONE_LIB_DIR at an AIOne install.",
    );

    println!(
        "cargo:rerun-if-changed={}",
        aione_dir.join("CMakeLists.txt").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        aione_dir.join("include").display()
    );
    println!("cargo:rerun-if-changed={}", aione_dir.join("src").display());
    println!("cargo:rerun-if-env-changed=AIONE_BUILD_DIR");

    // The nested vulkan-shaders-gen configure adds ~150 chars of its own,
    // so on Windows keep the build dir very short.
    let build_dir = env::var_os("AIONE_BUILD_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            if cfg!(windows) {
                PathBuf::from(format!(r"C:\ab\{}", config.to_lowercase()))
            } else {
                out_dir.join("aione-build")
            }
        });
    let install_dir = out_dir.join("aione-install");
    let ninja = find_ninja().unwrap_or_else(|| {
        panic!("Ninja not found. Run `winget install Ninja-build.Ninja` and open a NEW terminal.")
    });

    reset_if_generator_mismatch(&build_dir);

    let msvc_env = build_env(&ninja);

    let mut configure = Command::new("cmake");
    configure
        .arg("-G").arg("Ninja")
        .arg(format!("-DCMAKE_MAKE_PROGRAM={}", ninja.display()));
    if !msvc_env.is_empty() {
        configure.arg("-DCMAKE_C_COMPILER=cl");
        configure.arg("-DCMAKE_CXX_COMPILER=cl");
    }
    configure
        .arg("-S")
        .arg(&aione_dir)
        .arg("-B")
        .arg(&build_dir)
        .arg(format!("-DCMAKE_BUILD_TYPE={config}"))
        .arg(format!("-DCMAKE_INSTALL_PREFIX={}", install_dir.display()))
        .arg("-DAIONE_BUILD_SHARED_LIBS=ON")
        .arg("-DAIONE_DEPS_SHARED=OFF")
        .arg("-DAIONE_UWP=OFF")
        .arg("-DAIONE_ENABLE_OPENAI=OFF")
        .arg("-DAIONE_ENABLE_STABLE_DIFFUSION=OFF")
        .arg(format!(
            "-DAIONE_ENABLE_TTS={}",
            if enable_tts { "ON" } else { "OFF" }
        ))
        .arg("-DAIONE_VULKAN=ON")
        .arg("-DAIONE_OPENCL=OFF")
        .arg("-DAIONE_VERBOSE=OFF")
        .arg("-DLLAMA_BUILD_TESTS=OFF")
        .arg("-DLLAMA_BUILD_EXAMPLES=OFF")
        .arg("-DLLAMA_BUILD_TOOLS=OFF")
        .arg("-DLLAMA_BUILD_SERVER=OFF");
    apply_env(&mut configure, &msvc_env);
    run(&mut configure);

    let mut build = Command::new("cmake");
    build
        .arg("--build")
        .arg(&build_dir)
        .arg("--config")
        .arg(config)
        .arg("--target")
        .arg("AIOneCore");
    apply_env(&mut build, &msvc_env);
    run(&mut build);

    let mut install = Command::new("cmake");
    install
        .arg("--install")
        .arg(&build_dir)
        .arg("--config")
        .arg(config)
        .arg("--prefix")
        .arg(&install_dir);
    apply_env(&mut install, &msvc_env);
    run(&mut install);

    link_prebuilt(&install_dir, &out_dir)
        .unwrap_or_else(|error| panic!("Failed to stage AIOne runtime libraries: {error}"));
}

fn find_ninja() -> Option<PathBuf> {
    // 1. Ninja on PATH
    if Command::new("ninja")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        return Some(PathBuf::from("ninja"));
    }

    // 2. Ninja bundled with Visual Studio's CMake component
    let pf = env::var_os("ProgramFiles").map(PathBuf::from)?;
    let vs_root = pf.join("Microsoft Visual Studio");
    for version in fs::read_dir(&vs_root).ok()?.flatten() {
        for edition in fs::read_dir(version.path()).into_iter().flatten().flatten() {
            let candidate = edition
                .path()
                .join(r"Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe");
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

/// Environment (PATH, INCLUDE, LIB, ...) of the MSVC toolchain, found via the cc crate.
/// Empty on non-MSVC targets.
fn msvc_env() -> Vec<(OsString, OsString)> {
    let target = env::var("TARGET").unwrap_or_default();
    if !target.contains("msvc") {
        return Vec::new();
    }
    match cc::windows_registry::find_tool(&target, "cl.exe") {
        Some(tool) => tool.env().to_vec(),
        None => {
            println!("cargo:warning=Could not locate MSVC via cc; relying on the current environment");
            Vec::new()
        }
    }
}

fn build_env(ninja: &Path) -> Vec<(OsString, OsString)> {
    let mut vars = msvc_env();

    // Directory containing ninja.exe (empty if "ninja" is already on PATH)
    let ninja_dir = ninja.parent().filter(|p| !p.as_os_str().is_empty());

    if let Some(dir) = ninja_dir {
        // Find PATH in the cc-provided env (case-insensitive), or fall back to the process PATH
        let existing = vars
            .iter()
            .position(|(k, _)| k.to_string_lossy().eq_ignore_ascii_case("path"));
        let current = match existing {
            Some(i) => vars.remove(i).1,
            None => env::var_os("PATH").unwrap_or_default(),
        };
        let mut paths = vec![dir.to_path_buf()];
        paths.extend(env::split_paths(&current));
        vars.push((
            OsString::from("PATH"),
            env::join_paths(paths).expect("valid PATH"),
        ));
    }

    // Nested CMake projects (vulkan-shaders-gen) only see environment variables, not -D flags
    if !vars.is_empty() {
        vars.push((OsString::from("CC"), OsString::from("cl")));
        vars.push((OsString::from("CXX"), OsString::from("cl")));
    }
    vars.push((OsString::from("CMAKE_GENERATOR"), OsString::from("Ninja")));

    vars
}

fn apply_env(command: &mut Command, vars: &[(OsString, OsString)]) {
    for (key, value) in vars {
        command.env(key, value);
    }
}

fn reset_if_generator_mismatch(build_dir: &Path) {
    let cache = build_dir.join("CMakeCache.txt");
    let Ok(text) = fs::read_to_string(&cache) else {
        return;
    };
    let is_ninja = text
        .lines()
        .any(|l| l.starts_with("CMAKE_GENERATOR:INTERNAL=Ninja"));
    if !is_ninja {
        println!("cargo:warning=Removing stale CMake build dir (generator changed)");
        let _ = fs::remove_dir_all(build_dir);
    }
}

fn find_aione_dir(manifest_dir: &Path) -> Option<PathBuf> {
    if let Some(dir) = env::var_os("AIONE_DIR").map(PathBuf::from) {
        if is_aione_dir(&dir) {
            return Some(dir);
        }
    }

    [
        manifest_dir.join("../AIOne"),
        manifest_dir.join("../../AIOne"),
        manifest_dir.join("AIOne"),
    ]
    .into_iter()
    .find(|candidate| is_aione_dir(candidate))
}

fn is_aione_dir(dir: &Path) -> bool {
    dir.join("CMakeLists.txt").is_file() && dir.join("include").join("AIOne.h").is_file()
}

fn link_prebuilt(prefix: &Path, out_dir: &Path) -> std::io::Result<()> {
    let lib_dir = if prefix.join("lib").is_dir() {
        prefix.join("lib")
    } else {
        prefix.to_path_buf()
    };
    let bin_dir = if prefix.join("bin").is_dir() {
        prefix.join("bin")
    } else {
        prefix.to_path_buf()
    };

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=AIOneCore");

    let profile_dir = out_dir
        .ancestors()
        .nth(3)
        .map(Path::to_path_buf)
        .expect("cargo profile directory");

    copy_runtime_libraries(&bin_dir, &profile_dir)?;

    if bin_dir != lib_dir {
        copy_runtime_libraries(&lib_dir, &profile_dir)?;
    }

    Ok(())
}

fn copy_runtime_libraries(source_dir: &Path, destination_dir: &Path) -> std::io::Result<()> {
    for entry in fs::read_dir(source_dir)? {
        let entry = entry?;
        let source = entry.path();

        if !is_runtime_library(&source) {
            continue;
        }

        let Some(name) = source.file_name() else {
            continue;
        };

        let destination = destination_dir.join(name);
        fs::copy(&source, &destination)?;
    }

    Ok(())
}

fn is_runtime_library(path: &Path) -> bool {
    match path.extension().and_then(|extension| extension.to_str()) {
        Some("dll") | Some("so") | Some("dylib") => true,
        _ => path
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| name.contains(".so."))
            .unwrap_or(false),
    }
}

fn run(command: &mut Command) {
    let status = command
        .status()
        .unwrap_or_else(|error| panic!("Failed to run {command:?}: {error}"));

    if !status.success() {
        panic!("Command failed with {status}: {command:?}");
    }
}
