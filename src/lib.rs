//! Safe Rust bindings for the native AIOne C API.
//!
//! The crate wraps the opaque `aione_session` handle and exposes a small,
//! safe surface for loading models and streaming generation results.
//!
//! Enable the `serde` feature to derive [`serde::Serialize`] for
//! [`GenerateResult`] and [`GenerateStats`].

use std::ffi::{CStr, CString};
use std::fmt;
use std::os::raw::{c_char, c_float, c_int, c_uint, c_void};
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

unsafe extern "C" {
    fn aione_session_create() -> *mut AioneSession;
    fn aione_session_destroy(session: *mut AioneSession);
    fn aione_session_load_model(
        session: *mut AioneSession,
        path: *const c_char,
        on_progress: Option<unsafe extern "C" fn(c_float, *mut c_void)>,
        user_data: *mut c_void,
    ) -> c_int;
    fn aione_session_unload_model(session: *mut AioneSession);
    fn aione_session_is_loaded(session: *const AioneSession) -> c_int;
    fn aione_session_generate(
        session: *mut AioneSession,
        prompt: *const c_char,
        max_tokens: c_uint,
        callbacks: *const AioneGenerateCallbacks,
    ) -> AioneGenerateResult;
    fn aione_generate_result_free(result: *mut AioneGenerateResult);
    fn aione_last_error() -> *const c_char;
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
    fn aione_chat_update_metadata(
        folder: *const c_char,
        title: *const c_char,
        model: *const c_char,
        system_prompt: *const c_char,
    ) -> c_int;
    fn aione_chat_delete(folder: *const c_char) -> c_int;
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

#[derive(Default)]
pub struct GenerateCallbacks {
    pub on_token: Option<Box<dyn FnMut(&str)>>,
    pub on_token_reasoning: Option<Box<dyn FnMut(&str, bool)>>,
    pub on_input_eval: Option<Box<dyn FnMut(f32)>>,
    pub on_generation_start: Option<Box<dyn FnMut()>>,
    pub on_think_state_change: Option<Box<dyn FnMut(bool)>>,
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

    pub fn load_model<F>(&self, path: &str, on_progress: F) -> Result<()>
    where
        F: FnMut(f32),
    {
        let path = CString::new(path)
            .map_err(|_| AioneError::new("model path contains a null byte"))?;
        let mut context = ProgressContext {
            callback: on_progress,
        };

        let loaded = unsafe {
            aione_session_load_model(
                self.raw,
                path.as_ptr(),
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
        max_tokens: u32,
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
            user_data: (callbacks as *mut GenerateCallbacks).cast(),
        };

        let mut raw_result = unsafe {
            aione_session_generate(self.raw, prompt.as_ptr(), max_tokens, &raw_callbacks)
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
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe { aione_session_destroy(self.raw) };
    }
}

fn to_c_string(value: &str, label: &str) -> Result<CString> {
    CString::new(value).map_err(|_| AioneError::new(format!("{label} contains a null byte")))
}

#[cfg(feature = "serde")]
fn optional_c_string(value: Option<&str>, label: &str) -> Result<Option<CString>> {
    value.map(|value| to_c_string(value, label)).transpose()
}

#[cfg(feature = "serde")]
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
#[serde(rename_all = "camelCase")]
pub struct ChatMessage {
    pub id: u64,
    pub parent_id: u64,
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
    let folder = to_c_string(folder, "chat folder")?;
    let role = to_c_string(role, "role")?;
    let content = to_c_string(content, "content")?;

    let json = unsafe {
        take_json_string(aione_chat_append_json(
            folder.as_ptr(),
            role.as_ptr(),
            content.as_ptr(),
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
