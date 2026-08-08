import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "named_entity_recognition.main:app",
        host="0.0.0.0",
        port=8080,
        workers=1,
    )

