import os
import sys
import io
import uuid
import json
import base64
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core.speech_service import SpeechRecognitionService, StreamingWakeWordDetector
from core.audio_preprocessing import compute_mfcc, VAD, load_audio, save_audio

app = FastAPI(title="语音唤醒词检测与命令识别服务", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

speech_service = SpeechRecognitionService()

active_streams = {}


class WakeWordRegisterRequest(BaseModel):
    name: str


class CommandTrainRequest(BaseModel):
    name: str
    num_states: int = 5
    num_mixtures: int = 3


class ReloadModelsRequest(BaseModel):
    pass


@app.get("/api/status")
async def get_status():
    return JSONResponse(content=speech_service.get_system_status())


@app.get("/api/detection/history")
async def get_detection_history(limit: int = 100, wake_word: Optional[str] = None):
    history = speech_service.get_detection_history(limit=limit, wake_word_filter=wake_word)
    return JSONResponse(content={"history": history, "count": len(history)})


@app.get("/api/detection/stats")
async def get_detection_stats():
    stats = speech_service.get_detection_stats()
    return JSONResponse(content=stats)


@app.get("/api/wake-words")
async def get_wake_words():
    return JSONResponse(content={"wake_words": speech_service.get_wake_words()})


@app.post("/api/wake-words/{name}/register")
async def register_wake_word(name: str, files: List[UploadFile] = File(...)):
    if len(files) < 3:
        raise HTTPException(status_code=400, detail="至少需要3段音频样本")

    temp_files = []
    try:
        for file in files:
            contents = await file.read()
            temp_path = os.path.join(config.TEMP_DIR, f"{uuid.uuid4()}.wav")
            with open(temp_path, 'wb') as f:
                f.write(contents)
            temp_files.append(temp_path)

        success, message = speech_service.register_wake_word(name, temp_files)

        if success:
            return JSONResponse(content={"success": True, "message": message})
        else:
            raise HTTPException(status_code=400, detail=message)
    finally:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)


@app.post("/api/wake-words/{name}/add-sample")
async def add_wake_word_sample(name: str, file: UploadFile = File(...)):
    temp_path = os.path.join(config.TEMP_DIR, f"{uuid.uuid4()}.wav")
    try:
        contents = await file.read()
        with open(temp_path, 'wb') as f:
            f.write(contents)

        success, message = speech_service.add_wake_word_sample(name, temp_path)

        if success:
            return JSONResponse(content={"success": True, "message": message})
        else:
            raise HTTPException(status_code=400, detail=message)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.delete("/api/wake-words/{name}")
async def delete_wake_word(name: str):
    success, message = speech_service.delete_wake_word(name)
    if success:
        return JSONResponse(content={"success": True, "message": message})
    else:
        raise HTTPException(status_code=404, detail=message)


@app.get("/api/commands")
async def get_commands():
    return JSONResponse(content={"commands": speech_service.get_commands()})


@app.get("/api/commands/predefined")
async def get_predefined_commands():
    return JSONResponse(content={"commands": list(config.COMMANDS.keys())})


@app.post("/api/commands/{name}/train")
async def train_command(name: str, files: List[UploadFile] = File(...), num_states: int = 5, num_mixtures: int = 3):
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="至少需要2段音频样本")

    temp_files = []
    try:
        for file in files:
            contents = await file.read()
            temp_path = os.path.join(config.TEMP_DIR, f"{uuid.uuid4()}.wav")
            with open(temp_path, 'wb') as f:
                f.write(contents)
            temp_files.append(temp_path)

        success, message = speech_service.train_command(name, temp_files, num_states, num_mixtures)

        if success:
            return JSONResponse(content={"success": True, "message": message})
        else:
            raise HTTPException(status_code=400, detail=message)
    finally:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)


@app.delete("/api/commands/{name}")
async def delete_command(name: str):
    success, message = speech_service.delete_command(name)
    if success:
        return JSONResponse(content={"success": True, "message": message})
    else:
        raise HTTPException(status_code=404, detail=message)


@app.get("/api/digits")
async def get_digits():
    return JSONResponse(content={"digits": speech_service.get_digits()})


@app.post("/api/digits/{digit}/train")
async def train_digit(digit: str, files: List[UploadFile] = File(...), num_states: int = 3, num_mixtures: int = 2):
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="至少需要2段音频样本")

    temp_files = []
    try:
        for file in files:
            contents = await file.read()
            temp_path = os.path.join(config.TEMP_DIR, f"{uuid.uuid4()}.wav")
            with open(temp_path, 'wb') as f:
                f.write(contents)
            temp_files.append(temp_path)

        success, message = speech_service.train_digit(digit, temp_files, num_states, num_mixtures)

        if success:
            return JSONResponse(content={"success": True, "message": message})
        else:
            raise HTTPException(status_code=400, detail=message)
    finally:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)


@app.post("/api/detect/wake-word")
async def detect_wake_word(file: UploadFile = File(...)):
    temp_path = os.path.join(config.TEMP_DIR, f"{uuid.uuid4()}.wav")
    try:
        contents = await file.read()
        with open(temp_path, 'wb') as f:
            f.write(contents)

        result = speech_service.detect_wake_word_file(temp_path, return_details=True)

        return JSONResponse(content=result)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/recognize/command")
async def recognize_command(file: UploadFile = File(...)):
    temp_path = os.path.join(config.TEMP_DIR, f"{uuid.uuid4()}.wav")
    try:
        contents = await file.read()
        with open(temp_path, 'wb') as f:
            f.write(contents)

        result = speech_service.recognize_command_file(temp_path)

        return JSONResponse(content=result)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/recognize/full")
async def recognize_full(file: UploadFile = File(...)):
    temp_path = os.path.join(config.TEMP_DIR, f"{uuid.uuid4()}.wav")
    try:
        contents = await file.read()
        with open(temp_path, 'wb') as f:
            f.write(contents)

        result = speech_service.full_recognition_file(temp_path)

        return JSONResponse(content=result)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/api/speakers")
async def get_speakers():
    return JSONResponse(content={"speakers": speech_service.get_speakers()})


@app.post("/api/speakers/register")
async def register_speaker(name: str, files: List[UploadFile] = File(...)):
    if len(files) < config.SPEAKER_MIN_SAMPLES:
        raise HTTPException(status_code=400, detail=f"至少需要 {config.SPEAKER_MIN_SAMPLES} 段音频样本")

    temp_files = []
    try:
        for file in files:
            contents = await file.read()
            temp_path = os.path.join(config.TEMP_DIR, f"{uuid.uuid4()}.wav")
            with open(temp_path, 'wb') as f:
                f.write(contents)
            temp_files.append(temp_path)

        success, message = speech_service.register_speaker(name, temp_files)

        if success:
            return JSONResponse(content={"success": True, "message": message})
        else:
            raise HTTPException(status_code=400, detail=message)
    finally:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)


@app.delete("/api/speakers/{speaker_id}")
async def delete_speaker(speaker_id: str):
    success, message = speech_service.delete_speaker(speaker_id)
    if success:
        return JSONResponse(content={"success": True, "message": message})
    else:
        raise HTTPException(status_code=404, detail=message)


@app.post("/api/models/reload")
async def reload_models():
    speech_service.reload_all_models()
    return JSONResponse(content={"success": True, "status": speech_service.get_system_status()})


@app.get("/api/audio-analysis")
async def audio_analysis(file_path: Optional[str] = None):
    if file_path and os.path.exists(file_path):
        signal, sr = load_audio(file_path)
    else:
        duration = 2.0
        t = np.linspace(0, duration, int(config.SAMPLE_RATE * duration))
        signal = np.sin(2 * np.pi * 440 * t) * 0.5
        sr = config.SAMPLE_RATE

    vad = VAD()
    vad_result = vad.detect(signal)
    mfcc = compute_mfcc(signal)

    waveform_data = {
        "sample_rate": sr,
        "duration": len(signal) / sr,
        "samples": signal[::10].tolist()
    }

    mfcc_data = {
        "shape": list(mfcc.shape),
        "data": mfcc.T.tolist()
    }

    vad_data = {
        "speech_segments": vad_result["speech_segments"],
        "energy": vad_result["energy"].tolist(),
        "zcr": vad_result["zcr"].tolist(),
        "speech_mask": vad_result["speech_mask"].tolist()
    }

    return JSONResponse(content={
        "waveform": waveform_data,
        "mfcc": mfcc_data,
        "vad": vad_data
    })


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()

    stream_id = str(uuid.uuid4())
    stream_detector = StreamingWakeWordDetector(speech_service)
    active_streams[stream_id] = stream_detector

    try:
        await websocket.send_json({
            "type": "connected",
            "stream_id": stream_id
        })

        while True:
            data = await websocket.receive()

            if data["type"] == "websocket.disconnect":
                break

            if "bytes" in data:
                audio_bytes = data["bytes"]
                audio_data = np.frombuffer(audio_bytes, dtype=np.float32)

                results = stream_detector.process_audio(audio_data)

                for result in results:
                    await websocket.send_json({
                        "type": "detection_result",
                        "data": result
                    })

                if stream_detector.wake_word_detected:
                    command_window_samples = int(config.COMMAND_WINDOW_DURATION * config.SAMPLE_RATE)
                    if len(stream_detector.buffer) >= command_window_samples:
                        command_signal = stream_detector.buffer[-command_window_samples:]
                        command_result = speech_service.recognize_command(command_signal)

                        await websocket.send_json({
                            "type": "command_result",
                            "data": command_result
                        })

                        stream_detector.reset()

                confidence_history = stream_detector.get_confidence_history(max_points=50)
                await websocket.send_json({
                    "type": "confidence_update",
                    "data": confidence_history
                })

            elif "text" in data:
                try:
                    message = json.loads(data["text"])
                    if message.get("type") == "reset":
                        stream_detector.reset()
                        await websocket.send_json({
                            "type": "reset_ack",
                            "success": True
                        })
                    elif message.get("type") == "get_status":
                        await websocket.send_json({
                            "type": "status",
                            "data": {
                                "buffer_duration": stream_detector.get_buffer_duration(),
                                "wake_word_detected": stream_detector.wake_word_detected,
                                "detected_wake_word": stream_detector.detected_wake_word
                            }
                        })
                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        pass
    finally:
        if stream_id in active_streams:
            del active_streams[stream_id]


@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    await websocket.accept()

    audio_buffer = np.array([], dtype=np.float32)
    sample_rate = config.SAMPLE_RATE

    try:
        await websocket.send_json({"type": "connected"})

        while True:
            data = await websocket.receive()

            if data["type"] == "websocket.disconnect":
                break

            if "bytes" in data:
                audio_bytes = data["bytes"]
                chunk = np.frombuffer(audio_bytes, dtype=np.float32)
                audio_buffer = np.concatenate([audio_buffer, chunk])

                if len(audio_buffer) >= sample_rate * 0.5:
                    result = speech_service.full_recognition(audio_buffer)

                    waveform = {
                        "samples": audio_buffer[::10].tolist(),
                        "sample_rate": sample_rate,
                        "duration": len(audio_buffer) / sample_rate
                    }

                    vad = VAD()
                    vad_result = vad.detect(audio_buffer)
                    mfcc = compute_mfcc(audio_buffer)

                    await websocket.send_json({
                        "type": "analysis_update",
                        "data": {
                            "waveform": waveform,
                            "mfcc_shape": list(mfcc.shape),
                            "mfcc_data": mfcc.T.tolist(),
                            "vad_segments": vad_result["speech_segments"]
                        }
                    })

                    if result["wake_word_detected"]:
                        await websocket.send_json({
                            "type": "full_result",
                            "data": result
                        })
                        audio_buffer = np.array([], dtype=np.float32)

            elif "text" in data:
                try:
                    message = json.loads(data["text"])
                    if message.get("type") == "clear":
                        audio_buffer = np.array([], dtype=np.float32)
                        await websocket.send_json({"type": "cleared"})
                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        pass


def ensure_directories():
    directories = [
        config.MODELS_DIR,
        config.WAKE_WORDS_DIR,
        config.COMMANDS_DIR,
        config.DIGITS_DIR,
        config.SPEAKERS_DIR,
        config.SAMPLES_DIR,
        config.TEMP_DIR
    ]
    for d in directories:
        os.makedirs(d, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    uvicorn.run(app, host="0.0.0.0", port=8000)
