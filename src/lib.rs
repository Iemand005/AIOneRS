//! Safe Rust bindings for the native AIOne C API.
//!
//! The crate wraps the opaque `aione_session` handle and exposes a small,
//! safe surface for loading models and streaming generation results.
//!
//! Enable the `serde` feature to derive [`serde::Serialize`] for
//! [`GenerateResult`] and [`GenerateStats`].

use std::ffi::{CStr, CString};
use std::fmt;
use std::os::raw::{c_char, c_float, c_int, c_void};
use std::ptr;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub type Result<T> = std::result::Result<T, AioneError>;

#[repr(C)]
struct AioneSession {
    _private: [u8; 0],
}

#[repr(C)]
struct AioneGenerateCallbacks {
    on_token: Option<unsafe extern "C" fn(*const c_char, *mut c_void)>,
    on_token_reasoning: Option<unsafe extern "C" fn(*const c_char, c_int, *mut c_void)>,
    on_input_eval: Option<unsafe extern "C" fn(c_float, *mut c_void)>,
    on_generation_start: Option<unsafe extern "C" fn(*mut c_void)>,
    on_think_state_change: Option<unsafe extern "C" fn(c_int, *mut c_void)>,
    should_stop: Option<unsafe extern "C" fn(*mut c_void) -> c_int>,
    user_data: *mut c_void,
}

#[repr(C)]
struct AioneGenerateResult {
    content: *mut c_char,
    tokens_evaluated: u64,
    tokens_generated: u64,
    tokens_cached: u64,
    prompt_eval_ms: f64,
    generation_ms: f64,
    prompt_tokens_per_second: f64,
    tokens_per_second: f64,
    ok: c_int,
}

impl Default for AioneGenerateResult {
    fn default() -> Self {
        Self {
            content: ptr::null_mut(),
            tokens_evaluated: 0,
            tokens_generated: 0,
            tokens_cached: 0,
            prompt_eval_ms: 0.0,
            generation_ms: 0.0,
            prompt_tokens_per_second: 0.0,
            tokens_per_second: 0.0,
            ok: 0,
        }
    }
}

#[repr(C)]
struct AioneLoadModelOptions {
    no_host: c_int,
    context_length: u32,
}

#[repr(C)]
struct AioneTts {
    _private: [u8; 0],
}

#[repr(C)]
struct AioneTtsLoadOptions {
    thread_count: c_int,
    flash_attention: c_int,
    device: c_int,
    on_progress: Option<unsafe extern "C" fn(c_int, f32, *mut c_void)>,
    user_data: *mut c_void,
}

#[repr(C)]
struct AioneTtsVoice {
    reference_file: *const c_char,
    built_in_name: *const c_char,
    built_in_index: c_int,
}

#[repr(C)]
struct AioneTtsSynthesisOptions {
    output_type: c_int,
    voice: AioneTtsVoice,
    lang: *const c_char,
    top_k: c_int,
    top_p: f32,
    seed: u32,
    batch_size: c_int,
    max_tokens: c_int,
}

#[repr(C)]
struct AioneTtsSynthesisResult {
    sample_rate: c_int,
    samples: i64,
    audio_type: c_int,
    data: *mut c_char,
    data_size: u64,
    prompt_eval_ms: f64,
    generation_ms: f64,
    conversion_ms: f64,
    ok: c_int,
}

unsafe extern "C" {
    fn aione_session_create() -> *mut AioneSession;
    fn aione_session_destroy(session: *mut AioneSession);
    fn aione_session_load_model(
        session: *mut AioneSession,
        path: *const c_char,
        options: *const AioneLoadModelOptions,
        on_progress: Option<unsafe extern "C" fn(c_float, *mut c_void)>,
        user_data: *mut c_void,
    ) -> c_int;
    fn aione_session_unload_model(session: *mut AioneSession);
    fn aione_session_is_loaded(session: *const AioneSession) -> c_int;
    fn aione_session_generate(
        session: *mut AioneSession,
        prompt: *const c_char,
        options: *const GenerateOptions,
        callbacks: *const AioneGenerateCallbacks,
    ) -> AioneGenerateResult;
    fn aione_generate_result_free(result: *mut AioneGenerateResult);
    fn aione_format_chat_prompt_json(
        session: *mut AioneSession,
        messages_json: *const c_char,
        add_generation_prompt: c_int,
    ) -> *mut c_char;
    fn aione_last_error() -> *const c_char;
    fn aione_system_info_json() -> *mut c_char;
    fn aione_chat_list_json() -> *mut c_char;
    fn aione_chat_create_json(
        title: *const c_char,
        model: *const c_char,
        system_prompt: *const c_char,
    ) -> *mut c_char;
    fn aione_chat_load_json(folder: *const c_char) -> *mut c_char;
    fn aione_chat_append_json(
        folder: *const c_char,
        role: *const c_char,
        content: *const c_char,
    ) -> *mut c_char;
    fn aione_chat_append_with_parent_json(
        folder: *const c_char,
        role: *const c_char,
        content: *const c_char,
        parent_id: u64,
    ) -> *mut c_char;
    fn aione_chat_update_metadata(
        folder: *const c_char,
        title: *const c_char,
        model: *const c_char,
        system_prompt: *const c_char,
    ) -> c_int;
    fn aione_chat_delete(folder: *const c_char) -> c_int;
    fn aione_tts_create() -> *mut AioneTts;
    fn aione_tts_destroy(tts: *mut AioneTts);
    fn aione_tts_load_model(
        tts: *mut AioneTts,
        model_path: *const c_char,
        mmproj_path: *const c_char,
        options: *const AioneTtsLoadOptions,
    ) -> c_int;
    fn aione_tts_unload_model(tts: *mut AioneTts);
    fn aione_tts_is_loaded(tts: *const AioneTts) -> c_int;
    fn aione_tts_voices_json(tts: *mut AioneTts) -> *mut c_char;
    fn aione_tts_synthesize(
        tts: *mut AioneTts,
        prompt: *const c_char,
        options: *const AioneTtsSynthesisOptions,
        on_frame: Option<unsafe extern "C" fn(c_int, *mut c_void)>,
        user_data: *mut c_void,
    ) -> AioneTtsSynthesisResult;
    fn aione_tts_synthesis_result_free(result: *mut AioneTtsSynthesisResult);
    fn aione_tts_save_wav(
        tts: *mut AioneTts,
        filename: *const c_char,
        data: *const c_char,
        data_size: u64,
    ) -> c_int;
    fn aione_string_free(value: *mut c_char);
}

#[derive(Debug, Clone)]
pub struct AioneError {
    message: String,
}

impl AioneError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    fn last() -> Self {
        let pointer = unsafe { aione_last_error() };
        let message = if pointer.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(pointer) }
                .to_string_lossy()
                .into_owned()
        };

        if message.is_empty() {
            Self::new("AIOne operation failed")
        } else {
            Self::new(message)
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for AioneError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for AioneError {}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct GenerateStats {
    pub prompt_tokens: u64,
    pub evaluated_tokens: u64,
    pub cached_tokens: u64,
    pub generated_tokens: u64,
    pub prompt_eval_ms: f64,
    pub generation_ms: f64,
    pub prompt_tokens_per_second: f64,
    pub tokens_per_second: f64,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct GenerateResult {
    pub content: String,
    pub stats: GenerateStats,
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase", default))]
pub struct LoadOptions {
    pub no_host: bool,
    pub context_length: u32,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            no_host: false,
            context_length: 4096,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase", default))]
pub struct GenerateOptions {
    pub max_tokens: u32,
    pub min_p: f32,
    pub top_k: c_int,
    pub temperature: f32,
    pub seed: u32,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_tokens: 128,
            min_p: 0.05,
            top_k: 40,
            temperature: 0.8,
            seed: 0xFFFFFFFF,
        }
    }
}

#[derive(Default)]
pub struct GenerateCallbacks {
    pub on_token: Option<Box<dyn FnMut(&str)>>,
    pub on_token_reasoning: Option<Box<dyn FnMut(&str, bool)>>,
    pub on_input_eval: Option<Box<dyn FnMut(f32)>>,
    pub on_generation_start: Option<Box<dyn FnMut()>>,
    pub on_think_state_change: Option<Box<dyn FnMut(bool)>>,
    pub should_stop: Option<Box<dyn FnMut() -> bool>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub enum TtsDevice {
    Any,
    Cpu,
    Igpu,
    Dgpu,
    Accelerator,
}

impl Default for TtsDevice {
    fn default() -> Self {
        Self::Any
    }
}

impl TtsDevice {
    fn as_raw(self) -> c_int {
        match self {
            Self::Any => 0,
            Self::Cpu => 1,
            Self::Igpu => 2,
            Self::Dgpu => 3,
            Self::Accelerator => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub enum TtsAudioType {
    Pcm,
    Wav,
}

impl Default for TtsAudioType {
    fn default() -> Self {
        Self::Pcm
    }
}

impl TtsAudioType {
    fn as_raw(self) -> c_int {
        match self {
            Self::Pcm => 0,
            Self::Wav => 1,
        }
    }

    fn from_raw(value: c_int) -> Self {
        match value {
            1 => Self::Wav,
            _ => Self::Pcm,
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase", default))]
pub struct TtsLoadOptions {
    pub thread_count: i32,
    pub flash_attention: bool,
    pub device: TtsDevice,
}

impl Default for TtsLoadOptions {
    fn default() -> Self {
        Self {
            thread_count: 6,
            flash_attention: true,
            device: TtsDevice::Any,
        }
    }
}

#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase", default))]
pub struct TtsVoice {
    pub reference_file: Option<String>,
    pub built_in_name: Option<String>,
    pub built_in_index: i32,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase", default))]
pub struct TtsSynthesisOptions {
    pub output_type: TtsAudioType,
    pub voice: TtsVoice,
    pub lang: Option<String>,
    pub top_k: i32,
    pub top_p: f32,
    pub seed: u32,
    pub batch_size: i32,
    pub max_tokens: i32,
}

impl Default for TtsSynthesisOptions {
    fn default() -> Self {
        Self {
            output_type: TtsAudioType::Pcm,
            voice: TtsVoice::default(),
            lang: None,
            top_k: 50,
            top_p: 1.0,
            seed: 0xFFFF_FFFF,
            batch_size: 512,
            max_tokens: -1,
        }
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct TtsSynthesisStats {
    pub prompt_eval_ms: f64,
    pub generation_ms: f64,
    pub conversion_ms: f64,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct TtsSynthesisResult {
    pub sample_rate: i32,
    pub samples: i64,
    pub audio_type: TtsAudioType,
    pub audio: Vec<u8>,
    pub stats: TtsSynthesisStats,
}

#[derive(Default)]
pub struct TtsSynthesisCallbacks {
    pub on_frame: Option<Box<dyn FnMut(i32)>>,
}

unsafe extern "C" fn tts_frame_trampoline(frame_index: c_int, user_data: *mut c_void) {
    if user_data.is_null() {
        return;
    }

    let callbacks = unsafe { &mut *(user_data as *mut TtsSynthesisCallbacks) };
    if let Some(callback) = callbacks.on_frame.as_mut() {
        callback(frame_index);
    }
}

struct ProgressContext<F> {
    callback: F,
}

unsafe extern "C" fn progress_trampoline<F: FnMut(f32)>(
    progress: c_float,
    user_data: *mut c_void,
) {
    if user_data.is_null() {
        return;
    }

    let context = unsafe { &mut *(user_data as *mut ProgressContext<F>) };
    (context.callback)(progress);
}

unsafe extern "C" fn token_trampoline(token: *const c_char, user_data: *mut c_void) {
    if token.is_null() || user_data.is_null() {
        return;
    }

    let callbacks = unsafe { &mut *(user_data as *mut GenerateCallbacks) };
    if let Some(callback) = callbacks.on_token.as_mut() {
        let token = unsafe { CStr::from_ptr(token) }.to_string_lossy();
        callback(&token);
    }
}

unsafe extern "C" fn token_reasoning_trampoline(
    token: *const c_char,
    thinking: c_int,
    user_data: *mut c_void,
) {
    if token.is_null() || user_data.is_null() {
        return;
    }

    let callbacks = unsafe { &mut *(user_data as *mut GenerateCallbacks) };
    if let Some(callback) = callbacks.on_token_reasoning.as_mut() {
        let token = unsafe { CStr::from_ptr(token) }.to_string_lossy();
        callback(&token, thinking != 0);
    }
}

unsafe extern "C" fn input_eval_trampoline(progress: c_float, user_data: *mut c_void) {
    if user_data.is_null() {
        return;
    }

    let callbacks = unsafe { &mut *(user_data as *mut GenerateCallbacks) };
    if let Some(callback) = callbacks.on_input_eval.as_mut() {
        callback(progress);
    }
}

unsafe extern "C" fn generation_start_trampoline(user_data: *mut c_void) {
    if user_data.is_null() {
        return;
    }

    let callbacks = unsafe { &mut *(user_data as *mut GenerateCallbacks) };
    if let Some(callback) = callbacks.on_generation_start.as_mut() {
        callback();
    }
}

unsafe extern "C" fn should_stop_trampoline(user_data: *mut c_void) -> c_int {
    if user_data.is_null() {
        return 0;
    }

    let callbacks = unsafe { &mut *(user_data as *mut GenerateCallbacks) };
    callbacks
        .should_stop
        .as_mut()
        .is_some_and(|callback| callback()) as c_int
}

unsafe extern "C" fn think_state_trampoline(thinking: c_int, user_data: *mut c_void) {
    if user_data.is_null() {
        return;
    }

    let callbacks = unsafe { &mut *(user_data as *mut GenerateCallbacks) };
    if let Some(callback) = callbacks.on_think_state_change.as_mut() {
        callback(thinking != 0);
    }
}

pub struct Session {
    raw: *mut AioneSession,
}

unsafe impl Send for Session {}

impl Session {
    pub fn new() -> Result<Self> {
        let raw = unsafe { aione_session_create() };

        if raw.is_null() {
            Err(AioneError::last())
        } else {
            Ok(Self { raw })
        }
    }

    pub fn load_model<F>(&self, path: &str, options: LoadOptions, on_progress: F) -> Result<()>
    where
        F: FnMut(f32),
    {
        let path = CString::new(path)
            .map_err(|_| AioneError::new("model path contains a null byte"))?;
        let mut context = ProgressContext {
            callback: on_progress,
        };
        let options = AioneLoadModelOptions {
            no_host: options.no_host as c_int,
            context_length: options.context_length,
        };

        let loaded = unsafe {
            aione_session_load_model(
                self.raw,
                path.as_ptr(),
                &options,
                Some(progress_trampoline::<F>),
                (&mut context as *mut ProgressContext<F>).cast(),
            )
        };

        if loaded == 0 {
            Err(AioneError::last())
        } else {
            Ok(())
        }
    }

    pub fn unload_model(&self) {
        unsafe { aione_session_unload_model(self.raw) };
    }

    pub fn is_loaded(&self) -> bool {
        unsafe { aione_session_is_loaded(self.raw) != 0 }
    }

    pub fn generate(
        &self,
        prompt: &str,
        options: GenerateOptions,
        callbacks: &mut GenerateCallbacks,
    ) -> Result<GenerateResult> {
        let prompt = CString::new(prompt)
            .map_err(|_| AioneError::new("prompt contains a null byte"))?;

        let raw_callbacks = AioneGenerateCallbacks {
            on_token: callbacks
                .on_token
                .as_ref()
                .map(|_| token_trampoline as unsafe extern "C" fn(*const c_char, *mut c_void)),
            on_token_reasoning: callbacks.on_token_reasoning.as_ref().map(|_| {
                token_reasoning_trampoline
                    as unsafe extern "C" fn(*const c_char, c_int, *mut c_void)
            }),
            on_input_eval: callbacks
                .on_input_eval
                .as_ref()
                .map(|_| input_eval_trampoline as unsafe extern "C" fn(c_float, *mut c_void)),
            on_generation_start: callbacks.on_generation_start.as_ref().map(|_| {
                generation_start_trampoline as unsafe extern "C" fn(*mut c_void)
            }),
            on_think_state_change: callbacks.on_think_state_change.as_ref().map(|_| {
                think_state_trampoline as unsafe extern "C" fn(c_int, *mut c_void)
            }),
            should_stop: callbacks
                .should_stop
                .as_ref()
                .map(|_| should_stop_trampoline as unsafe extern "C" fn(*mut c_void) -> c_int),
            user_data: (callbacks as *mut GenerateCallbacks).cast(),
        };

        let mut raw_result = unsafe {
            aione_session_generate(
                self.raw,
                prompt.as_ptr(),
                &options,
                &raw_callbacks,
            )
        };

        if raw_result.ok == 0 {
            let error = AioneError::last();
            unsafe { aione_generate_result_free(&mut raw_result) };
            return Err(error);
        }

        let content = if raw_result.content.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(raw_result.content) }
                .to_string_lossy()
                .into_owned()
        };

        let result = GenerateResult {
            content,
            stats: GenerateStats {
                prompt_tokens: raw_result.tokens_evaluated + raw_result.tokens_cached,
                evaluated_tokens: raw_result.tokens_evaluated,
                cached_tokens: raw_result.tokens_cached,
                generated_tokens: raw_result.tokens_generated,
                prompt_eval_ms: raw_result.prompt_eval_ms,
                generation_ms: raw_result.generation_ms,
                prompt_tokens_per_second: raw_result.prompt_tokens_per_second,
                tokens_per_second: raw_result.tokens_per_second,
            },
        };

        unsafe { aione_generate_result_free(&mut raw_result) };
        Ok(result)
    }

    #[cfg(feature = "serde")]
    pub fn format_chat_prompt(
        &mut self,
        messages: &[ChatPromptMessage],
        add_generation_prompt: bool,
    ) -> Result<String> {
        let json = serde_json::to_string(messages)
            .map_err(|error| AioneError::new(format!("failed to encode messages: {error}")))?;
        let json = to_c_string(&json, "messages")?;

        unsafe {
            take_json_string(aione_format_chat_prompt_json(
                self.raw,
                json.as_ptr(),
                add_generation_prompt as c_int,
            ))
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe { aione_session_destroy(self.raw) };
    }
}

pub struct Tts {
    raw: *mut AioneTts,
}

unsafe impl Send for Tts {}

impl Tts {
    pub fn new() -> Result<Self> {
        let raw = unsafe { aione_tts_create() };

        if raw.is_null() {
            Err(AioneError::last())
        } else {
            Ok(Self { raw })
        }
    }

    pub fn load_model(
        &self,
        model_path: &str,
        mmproj_path: &str,
        options: TtsLoadOptions,
    ) -> Result<()> {
        let model_path = to_c_string(model_path, "model path")?;
        let mmproj_path = to_c_string(mmproj_path, "mmproj path")?;
        let options = AioneTtsLoadOptions {
            thread_count: options.thread_count,
            flash_attention: options.flash_attention as c_int,
            device: options.device.as_raw(),
            on_progress: None,
            user_data: ptr::null_mut(),
        };

        let loaded = unsafe {
            aione_tts_load_model(
                self.raw,
                model_path.as_ptr(),
                mmproj_path.as_ptr(),
                &options,
            )
        };

        if loaded == 0 {
            Err(AioneError::last())
        } else {
            Ok(())
        }
    }

    pub fn unload_model(&self) {
        unsafe { aione_tts_unload_model(self.raw) };
    }

    pub fn is_loaded(&self) -> bool {
        unsafe { aione_tts_is_loaded(self.raw) != 0 }
    }

    pub fn voices_json(&self) -> Result<String> {
        unsafe { take_json_string(aione_tts_voices_json(self.raw)) }
    }

    #[cfg(feature = "serde")]
    pub fn voices(&self) -> Result<Vec<String>> {
        let json = self.voices_json()?;
        parse_json(&json, "TTS voices")
    }

    pub fn synthesize(
        &self,
        prompt: &str,
        options: TtsSynthesisOptions,
        callbacks: &mut TtsSynthesisCallbacks,
    ) -> Result<TtsSynthesisResult> {
        let prompt =
            CString::new(prompt).map_err(|_| AioneError::new("prompt contains a null byte"))?;
        let reference_file =
            optional_c_string(options.voice.reference_file.as_deref(), "reference file")?;
        let built_in_name =
            optional_c_string(options.voice.built_in_name.as_deref(), "built-in voice")?;
        let lang = optional_c_string(options.lang.as_deref(), "language")?;

        let raw_options = AioneTtsSynthesisOptions {
            output_type: options.output_type.as_raw(),
            voice: AioneTtsVoice {
                reference_file: reference_file.as_ref().map_or(ptr::null(), |value| value.as_ptr()),
                built_in_name: built_in_name.as_ref().map_or(ptr::null(), |value| value.as_ptr()),
                built_in_index: options.voice.built_in_index,
            },
            lang: lang.as_ref().map_or(ptr::null(), |value| value.as_ptr()),
            top_k: options.top_k,
            top_p: options.top_p,
            seed: options.seed,
            batch_size: options.batch_size,
            max_tokens: options.max_tokens,
        };

        let on_frame = callbacks
            .on_frame
            .as_ref()
            .map(|_| tts_frame_trampoline as unsafe extern "C" fn(c_int, *mut c_void));
        let user_data: *mut c_void = (callbacks as *mut TtsSynthesisCallbacks).cast();

        let mut raw = unsafe {
            aione_tts_synthesize(
                self.raw,
                prompt.as_ptr(),
                &raw_options,
                on_frame,
                user_data,
            )
        };

        if raw.ok == 0 {
            let error = AioneError::last();
            unsafe { aione_tts_synthesis_result_free(&mut raw) };
            return Err(error);
        }

        let audio = if raw.data.is_null() || raw.data_size == 0 {
            Vec::new()
        } else {
            let size = usize::try_from(raw.data_size)
                .map_err(|_| AioneError::new("audio buffer is too large"))?;
            unsafe { std::slice::from_raw_parts(raw.data as *const u8, size) }.to_vec()
        };

        let result = TtsSynthesisResult {
            sample_rate: raw.sample_rate,
            samples: raw.samples,
            audio_type: TtsAudioType::from_raw(raw.audio_type),
            audio,
            stats: TtsSynthesisStats {
                prompt_eval_ms: raw.prompt_eval_ms,
                generation_ms: raw.generation_ms,
                conversion_ms: raw.conversion_ms,
            },
        };

        unsafe { aione_tts_synthesis_result_free(&mut raw) };
        Ok(result)
    }

    pub fn save_wav(&self, audio: &TtsSynthesisResult, filename: &str) -> Result<()> {
        if audio.audio_type != TtsAudioType::Wav {
            return Err(AioneError::new("audio is not WAV encoded"));
        }

        let filename = to_c_string(filename, "filename")?;
        let data_size = u64::try_from(audio.audio.len())
            .map_err(|_| AioneError::new("audio buffer is too large"))?;

        let saved = unsafe {
            aione_tts_save_wav(
                self.raw,
                filename.as_ptr(),
                audio.audio.as_ptr() as *const c_char,
                data_size,
            )
        };

        if saved == 0 {
            Err(AioneError::last())
        } else {
            Ok(())
        }
    }
}

impl Drop for Tts {
    fn drop(&mut self) {
        unsafe { aione_tts_destroy(self.raw) };
    }
}

fn to_c_string(value: &str, label: &str) -> Result<CString> {
    CString::new(value).map_err(|_| AioneError::new(format!("{label} contains a null byte")))
}

fn optional_c_string(value: Option<&str>, label: &str) -> Result<Option<CString>> {
    value.map(|value| to_c_string(value, label)).transpose()
}

unsafe fn take_json_string(pointer: *mut c_char) -> Result<String> {
    if pointer.is_null() {
        return Err(AioneError::last());
    }

    let value = unsafe { CStr::from_ptr(pointer) }.to_string_lossy().into_owned();
    unsafe { aione_string_free(pointer) };
    Ok(value)
}

#[cfg(feature = "serde")]
fn parse_json<T>(json: &str, label: &str) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    serde_json::from_str(json)
        .map_err(|error| AioneError::new(format!("invalid {label} JSON: {error}")))
}

#[cfg(feature = "serde")]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatMetadata {
    pub folder: String,
    pub title: String,
    pub model: String,
    pub system_prompt: String,
    pub created: i64,
    pub updated: i64,
}

#[cfg(feature = "serde")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatPromptMessage {
    pub role: String,
    pub content: String,
}

#[cfg(feature = "serde")]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatMessage {
    pub id: String,
    pub parent_id: String,
    pub role: String,
    pub content: String,
    pub creation_time: i64,
    pub finish_time: i64,
}

#[cfg(feature = "serde")]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatData {
    pub folder: String,
    pub title: String,
    pub model: String,
    pub system_prompt: String,
    pub created: i64,
    pub updated: i64,
    pub messages: Vec<ChatMessage>,
    pub active_path: Vec<String>,
}

#[cfg(feature = "serde")]
pub fn system_info() -> Result<serde_json::Value> {
    let json = unsafe { take_json_string(aione_system_info_json()) }?;
    parse_json(&json, "system info")
}

#[cfg(feature = "serde")]
pub fn list_chats() -> Result<Vec<ChatMetadata>> {
    let json = unsafe { take_json_string(aione_chat_list_json()) }?;
    parse_json(&json, "chat list")
}

#[cfg(feature = "serde")]
pub fn create_chat(title: &str, model: &str, system_prompt: &str) -> Result<ChatMetadata> {
    let title = to_c_string(title, "title")?;
    let model = to_c_string(model, "model")?;
    let system_prompt = to_c_string(system_prompt, "system prompt")?;

    let json = unsafe {
        take_json_string(aione_chat_create_json(
            title.as_ptr(),
            model.as_ptr(),
            system_prompt.as_ptr(),
        ))
    }?;

    parse_json(&json, "chat metadata")
}

#[cfg(feature = "serde")]
pub fn load_chat(folder: &str) -> Result<ChatData> {
    let folder = to_c_string(folder, "chat folder")?;
    let json = unsafe { take_json_string(aione_chat_load_json(folder.as_ptr())) }?;
    parse_json(&json, "chat")
}

#[cfg(feature = "serde")]
pub fn append_message(folder: &str, role: &str, content: &str) -> Result<ChatMessage> {
    append_message_with_parent(folder, role, content, 0)
}

#[cfg(feature = "serde")]
pub fn append_message_with_parent(
    folder: &str,
    role: &str,
    content: &str,
    parent_id: u64,
) -> Result<ChatMessage> {
    let folder = to_c_string(folder, "chat folder")?;
    let role = to_c_string(role, "role")?;
    let content = to_c_string(content, "content")?;

    let json = unsafe {
        take_json_string(aione_chat_append_with_parent_json(
            folder.as_ptr(),
            role.as_ptr(),
            content.as_ptr(),
            parent_id,
        ))
    }?;

    parse_json(&json, "chat message")
}

#[cfg(feature = "serde")]
pub fn update_chat_metadata(
    folder: &str,
    title: Option<&str>,
    model: Option<&str>,
    system_prompt: Option<&str>,
) -> Result<()> {
    let folder = to_c_string(folder, "chat folder")?;
    let title = optional_c_string(title, "title")?;
    let model = optional_c_string(model, "model")?;
    let system_prompt = optional_c_string(system_prompt, "system prompt")?;

    let updated = unsafe {
        aione_chat_update_metadata(
            folder.as_ptr(),
            title.as_ref().map_or(ptr::null(), |value| value.as_ptr()),
            model.as_ref().map_or(ptr::null(), |value| value.as_ptr()),
            system_prompt
                .as_ref()
                .map_or(ptr::null(), |value| value.as_ptr()),
        )
    };

    if updated == 0 {
        Err(AioneError::last())
    } else {
        Ok(())
    }
}

#[cfg(feature = "serde")]
pub fn delete_chat(folder: &str) -> Result<()> {
    let folder = to_c_string(folder, "chat folder")?;

    if unsafe { aione_chat_delete(folder.as_ptr()) } == 0 {
        Err(AioneError::last())
    } else {
        Ok(())
    }
}
