from named_entity_recognition_api.models.engine_descriptor import EngineDescriptor

from named_entity_recognition.runtime import Runtime


def engine_descriptor(runtime: Runtime) -> EngineDescriptor:
    engine = runtime.engine
    return EngineDescriptor(
        name=engine.name,
        version=engine.version,
        model=engine.model_name,
        model_revision=engine.model_revision,
        device=engine.device,
        available=engine.ready,
    )

