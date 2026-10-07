/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.forwardmeasure.nlp.ner.client;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.forwardmeasure.nlp.ner.client.api.CapabilitiesApi;
import com.forwardmeasure.nlp.ner.client.api.ExtractionsApi;
import com.forwardmeasure.nlp.ner.client.api.JobsApi;
import com.forwardmeasure.nlp.ner.client.api.ProfilesApi;
import com.forwardmeasure.nlp.ner.client.model.NerExtractionRequest;
import com.forwardmeasure.nlp.ner.client.model.NerJobStatus;
import com.forwardmeasure.nlp.ner.client.model.SubmitNerJobRequest;
import com.forwardmeasure.nlp.ner.client.model.TextSegment;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

/** Wire contracts exercise generated code without editing generated Java or testing boilerplate. */
class NerClientHttpContractTest {
  private static final UUID ID = UUID.fromString("11111111-2222-3333-4444-555555555555");
  private static final String BASE = "/api/v1/named-entity-recognition";
  private static final String CREATED = "2026-10-07T12:34:56.123456789Z";
  private static final String EXTRACTION =
      """
      {"extraction_id":"11111111-2222-3333-4444-555555555555",
       "request_id":"caller-1","mentions":[{"segment_id":"page-1","label":"person",
       "text":"José","start":0,"end":4,"absolute_start":4294967296,
       "absolute_end":4294967300,"confidence":0.95}],
       "engines":[{"name":"gliner","model":"test-model","available":true}],
       "created_at":"2026-10-07T12:34:56.123456789Z","processing_time_ms":12}
      """;
  private static final String JOB =
      """
      {"job_id":"11111111-2222-3333-4444-555555555555","status":"completed",
       "result_available":true,"created_at":"2026-10-07T12:34:56.123456789Z"}
      """;
  private static final String PROFILE =
      """
      {"name":"general","version":"1","labels":["person"],"engines":["gliner"],
       "strategy":"single_engine","threshold":0.5,"allow_nested_entities":true}
      """;

  @Test
  void extractionPreservesUnicodeOffsetsHeadersAndNanosecondTimestamps() throws Exception {
    var received = new AtomicReference<String>();
    var observed = new AtomicInteger();
    var server =
        server(
            exchange -> {
              assertEquals("POST", exchange.getRequestMethod());
              assertEquals(BASE + "/extractions", exchange.getRequestURI().getPath());
              assertEquals(
                  "application/json", exchange.getRequestHeaders().getFirst("Content-Type"));
              assertEquals("trace-1", exchange.getRequestHeaders().getFirst("X-Correlation-ID"));
              assertEquals(
                  "Bearer test-token", exchange.getRequestHeaders().getFirst("Authorization"));
              received.set(
                  new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
              respond(exchange, 200, EXTRACTION, true);
            });
    try {
      var client =
          client(server)
              .setReadTimeout(Duration.ofSeconds(5))
              .setConnectTimeout(Duration.ofSeconds(2))
              .setRequestInterceptor(
                  builder -> builder.header("Authorization", "Bearer test-token"))
              .setResponseInterceptor(response -> observed.incrementAndGet());
      var response =
          new ExtractionsApi(client)
              .extractEntitiesWithHttpInfo(request(), Map.of("X-Correlation-ID", "trace-1"));
      assertEquals(200, response.getStatusCode());
      var result = response.getData();
      assertEquals(ID, result.getExtractionId());
      assertEquals(Instant.parse(CREATED), result.getCreatedAt().toInstant());
      assertEquals("José", result.getMentions().getFirst().getText());
      assertEquals(4_294_967_296L, result.getMentions().getFirst().getAbsoluteStart());
      assertEquals(4_294_967_300L, result.getMentions().getFirst().getAbsoluteEnd());
      assertEquals(0.95, result.getMentions().getFirst().getConfidence());
      assertEquals("test-model", result.getEngines().getFirst().getModel());
      assertEquals(1, observed.get());
      var body = client.getObjectMapper().readTree(received.get());
      assertEquals("caller-1", body.path("request_id").asText());
      assertEquals("José works here", body.path("segments").get(0).path("text").asText());
      assertEquals(4_294_967_296L, body.path("segments").get(0).path("base_offset").asLong());
      assertEquals("person", body.path("labels").get(0).asText());
    } finally {
      server.stop(0);
    }
  }

  @Test
  void jobsProfilesAndCapabilitiesUseContractPathsAndDecodeTheirModels() throws Exception {
    var paths = new CopyOnWriteArrayList<String>();
    var server =
        server(
            exchange -> {
              String path = exchange.getRequestURI().getRawPath();
              paths.add(exchange.getRequestMethod() + " " + path);
              String body;
              if (path.endsWith("/result"))
                body = "{\"job\":" + JOB + ",\"result\":" + EXTRACTION + "}";
              else if (path.contains("/jobs")) {
                if (exchange.getRequestMethod().equals("POST")) {
                  var input =
                      ApiClient.createDefaultObjectMapper().readTree(exchange.getRequestBody());
                  assertEquals("caller-1", input.path("extraction").path("request_id").asText());
                }
                body = JOB;
              } else if (path.endsWith("/profiles")) body = "{\"profiles\":[" + PROFILE + "]}";
              else if (path.contains("/profiles/")) body = PROFILE;
              else
                body =
                    """
                    {"service_version":"1.1.0","synchronous_extraction":true,"asynchronous_jobs":true,
                    "limits":{"max_segments":100,"max_segment_characters":100000,"max_total_characters":1000000,"max_labels":100},
                    "engines":[],"ensemble_strategies":["single_engine","union"]}
                    """;
              respond(
                  exchange, exchange.getRequestMethod().equals("POST") ? 202 : 200, body, false);
            });
    try {
      var client = client(server);
      var jobs = new JobsApi(client);
      var submitted =
          jobs.submitNerJobWithHttpInfo(new SubmitNerJobRequest().extraction(request()));
      assertEquals(202, submitted.getStatusCode());
      assertEquals(ID, submitted.getData().getJobId());
      assertEquals(NerJobStatus.COMPLETED, jobs.getNerJob(ID).getStatus());
      assertEquals(ID, jobs.getNerJobResult(ID).getResult().getExtractionId());
      var profiles = new ProfilesApi(client);
      assertEquals("general", profiles.listNerProfiles().getProfiles().getFirst().getName());
      assertEquals("1", profiles.getNerProfile("name /+é").getVersion());
      var capabilities = new CapabilitiesApi(client).getNerCapabilities();
      assertEquals(100, capabilities.getLimits().getMaxSegments());
      assertTrue(capabilities.getAsynchronousJobs());
      assertEquals(
          List.of(
              "POST " + BASE + "/jobs",
              "GET " + BASE + "/jobs/" + ID,
              "GET " + BASE + "/jobs/" + ID + "/result",
              "GET " + BASE + "/profiles",
              "GET " + BASE + "/profiles/name%20%2F%2B%C3%A9",
              "GET " + BASE + "/capabilities"),
          paths);
    } finally {
      server.stop(0);
    }
  }

  @Test
  void serviceFailuresRetainStatusHeadersAndBodyForRetryDecisions() throws Exception {
    var status = new AtomicInteger(400);
    var server =
        server(
            exchange -> {
              exchange.getRequestBody().readAllBytes();
              exchange.getResponseHeaders().set("Retry-After", "3");
              respond(
                  exchange,
                  status.get(),
                  status.get() == 503
                      ? ""
                      : "{\"error_code\":\"controlled\",\"message\":\"retry later\"}",
                  false);
            });
    try {
      var client = client(server);
      List<org.junit.jupiter.api.function.Executable> operations =
          List.of(
              () -> new ExtractionsApi(client).extractEntities(request()),
              () ->
                  new JobsApi(client).submitNerJob(new SubmitNerJobRequest().extraction(request())),
              () -> new JobsApi(client).getNerJob(ID),
              () -> new JobsApi(client).getNerJobResult(ID),
              () -> new ProfilesApi(client).getNerProfile("missing"),
              () -> new ProfilesApi(client).listNerProfiles(),
              () -> new CapabilitiesApi(client).getNerCapabilities());
      for (int code : new int[] {400, 401, 403, 404, 409, 429, 503}) {
        status.set(code);
        for (var operation : operations) {
          var failure = assertThrows(ApiException.class, operation);
          assertEquals(code, failure.getCode());
          assertEquals("3", failure.getResponseHeaders().firstValue("Retry-After").orElseThrow());
          assertTrue(failure.getMessage().contains(code == 503 ? "[no body]" : "controlled"));
          assertNotNull(failure.getResponseBody());
        }
      }
    } finally {
      server.stop(0);
    }
  }

  @Test
  void requiredArgumentsFailBeforeNetworkAndInterruptedCallsRestoreTheFlag() throws Exception {
    var calls = new AtomicInteger();
    var server =
        server(
            exchange -> {
              calls.incrementAndGet();
              respond(exchange, 200, JOB, false);
            });
    try {
      var client = client(server);
      List<org.junit.jupiter.api.function.Executable> invalid =
          List.of(
              () -> new ExtractionsApi(client).extractEntities(null),
              () -> new JobsApi(client).submitNerJob(null),
              () -> new JobsApi(client).getNerJob(null),
              () -> new JobsApi(client).getNerJobResult(null),
              () -> new ProfilesApi(client).getNerProfile(null));
      for (var operation : invalid)
        assertEquals(400, assertThrows(ApiException.class, operation).getCode());
      assertEquals(0, calls.get());
      var jobs = new JobsApi(client);
      Thread.currentThread().interrupt();
      try {
        var failure = assertThrows(ApiException.class, () -> jobs.getNerJob(ID));
        assertInstanceOf(InterruptedException.class, failure.getCause());
        assertTrue(Thread.currentThread().isInterrupted());
      } finally {
        Thread.interrupted();
      }
      assertFalse(Thread.currentThread().isInterrupted());
    } finally {
      server.stop(0);
    }
  }

  private static NerExtractionRequest request() {
    return new NerExtractionRequest()
        .requestId("caller-1")
        .labels(Set.of("person"))
        .segments(
            List.of(
                new TextSegment()
                    .segmentId("page-1")
                    .text("José works here")
                    .baseOffset(4_294_967_296L)
                    .sourceRef("source-1")));
  }

  private static ApiClient client(HttpServer server) {
    var client = new ApiClient();
    client.updateBaseUri("http://127.0.0.1:" + server.getAddress().getPort() + BASE);
    return client.setReadTimeout(Duration.ofSeconds(5));
  }

  private static HttpServer server(com.sun.net.httpserver.HttpHandler handler) throws Exception {
    var server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
    server.createContext("/", handler);
    server.start();
    return server;
  }

  private static void respond(HttpExchange exchange, int status, String body, boolean gzip)
      throws java.io.IOException {
    byte[] payload = body.getBytes(StandardCharsets.UTF_8);
    exchange.getResponseHeaders().set("Content-Type", "application/json");
    if (gzip) {
      var bytes = new java.io.ByteArrayOutputStream();
      try (var compressed = new java.util.zip.GZIPOutputStream(bytes)) {
        compressed.write(payload);
      }
      payload = bytes.toByteArray();
      exchange.getResponseHeaders().set("Content-Encoding", "gzip");
    }
    exchange.sendResponseHeaders(status, payload.length == 0 ? -1 : payload.length);
    if (payload.length > 0) exchange.getResponseBody().write(payload);
    exchange.close();
  }
}
