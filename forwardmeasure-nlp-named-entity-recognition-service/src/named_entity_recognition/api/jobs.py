from __future__ import annotations

from uuid import UUID

from fastapi import HTTPException
from named_entity_recognition_api.apis.jobs_api_base import BaseJobsApi
from named_entity_recognition_api.models.ner_job_response import NerJobResponse
from named_entity_recognition_api.models.ner_job_result import NerJobResult
from named_entity_recognition_api.models.submit_ner_job_request import SubmitNerJobRequest


class JobsApi(BaseJobsApi):
    async def submit_ner_job(self, submit_ner_job_request: SubmitNerJobRequest) -> NerJobResponse:
        raise HTTPException(
            status_code=501,
            detail="Durable asynchronous NER jobs are not enabled in the initial deployment",
        )

    async def get_ner_job(self, job_id: UUID) -> NerJobResponse:
        raise HTTPException(
            status_code=501,
            detail="Durable asynchronous NER jobs are not enabled in the initial deployment",
        )

    async def get_ner_job_result(self, job_id: UUID) -> NerJobResult:
        raise HTTPException(
            status_code=501,
            detail="Durable asynchronous NER jobs are not enabled in the initial deployment",
        )

