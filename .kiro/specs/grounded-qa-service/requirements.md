# Requirements Document

## Introduction

Grounded QA Service is a question-answering service that retrieves passages from a license-clear document collection, has a local open-weight instruction model answer **only** from those retrieved passages while citing passage IDs, and automatically measures answer quality so that every change is evaluated before it ships. The project is portfolio evidence for AI-engineer and ML-engineer roles, so every reported number must be reproducible from a script in this repository, and every third-party dependency must be proven on the target machine and license-checked.

The service reuses an existing retrieval library, `rag-lite-tfidf-eval` (TF-IDF, BM25, dense embeddings, hybrid fusion, cross-encoder reranking, chunking with citation IDs, IR metrics, FastAPI, Docker, and CI), as its retrieval layer rather than rewriting that functionality. A local open-weight instruction model is served on an NVIDIA RTX 3080 through an OpenAI-compatible server; an optional hosted API model may be used under an owner-set monthly spend cap.

The work is organized into milestones M0 through M5, each ending in an automated milestone gate that replaces an owner review stop. M0 proves the environment and local model server with a smoke test. M1 delivers the corpus, the evaluation set, and the reproduced retrieval baseline. M2 delivers answer generation with citations, validation, and refusal. M3 delivers the evaluation harness including an LLM judge checked against owner labels. M4 delivers configuration experiments, the generated results table, and the injection test. M5 delivers the FastAPI service, Docker Compose orchestration, the CI regression gate, and the README.

This document defines the requirements only. Model selection, prompt wording, retrieval-layer integration mechanics, and other implementation details are deferred to the design phase.

## Glossary

- **System**: The complete Grounded QA Service, comprising the retrieval layer, the answering pipeline, the guardrails, the evaluation harness, the experiments, the service, and the continuous-integration gate.
- **Rag_Lite**: The existing `rag-lite-tfidf-eval` library reused as the retrieval layer, providing TF-IDF, BM25, dense embeddings, hybrid fusion, cross-encoder reranking, chunking with citation IDs, and IR metrics.
- **Retrieval_Layer**: The component, built on Rag_Lite, that returns a ranked list of passages for a query.
- **Answering_Pipeline**: The component that takes a query, invokes the Retrieval_Layer, builds a prompt from the retrieved passages, and produces an Answer.
- **Answer**: The generated response to a query, consisting of answer text and zero or more Passage_ID citations.
- **Passage**: A chunk of corpus text with a stable, unique Passage_ID assigned at chunking time.
- **Passage_ID**: The stable, unique identifier of a Passage, used both by the Retrieval_Layer and as a citation in an Answer.
- **Corpus**: The license-clear document collection the System answers from.
- **Evaluation_Set**: The set of questions, each with gold supporting Passage_IDs and, where applicable, a gold answer, used to evaluate the System.
- **Unanswerable_Question**: An Evaluation_Set question whose answer is not contained in the Corpus and for which the correct System behavior is a Refusal.
- **Refusal**: The System response "not found in the documents" issued when retrieval confidence is below the configured threshold or when the Corpus does not contain the answer.
- **Local_Model**: The local open-weight instruction model served on the GPU through an OpenAI-compatible server.
- **Model_Server**: The OpenAI-compatible server process that serves the Local_Model on the NVIDIA RTX 3080.
- **Hosted_Model**: The optional hosted API model used under the Spend_Cap.
- **Spend_Cap**: The owner-set monthly spend limit, expressed in United States dollars, for the Hosted_Model.
- **LLM_Judge**: The model-based evaluator that scores an Answer's faithfulness to its cited Passages.
- **Faithfulness**: The property that every claim in an Answer is supported by the Passages the Answer cites.
- **Citation_Validation**: The step that rejects an Answer citing any Passage_ID not present in the retrieved set for that query.
- **Injection_Test**: The test that measures how often the Local_Model follows instructions embedded inside Corpus Passages.
- **Configuration**: A named combination of retrieval mode and model used in an Experiment (for example, BM25-only, or hybrid-plus-reranking with a larger model).
- **Experiment**: A measured comparison of Configurations on quality, latency, and cost.
- **Measured_Value**: A number produced by a script in this repository, recorded with its generating command, seed, and full commit hash.
- **Results_Table**: The generated file containing the measured Experiment results, including confidence intervals.
- **ENVIRONMENT_Doc**: The file `docs/ENVIRONMENT.md` recording exact tool, model, and dependency versions proven on the machine.
- **THIRD_PARTY_Doc**: The file `THIRD_PARTY.md` recording each third-party package, model, and asset and its license.
- **Baseline_File**: The stored file recording the baseline Measured_Values that the CI regression gate compares against.
- **CI_Pipeline**: The GitHub Actions continuous-integration pipeline.

## Requirements

### Requirement 0: Governing non-functional requirements

**User Story:** As the repository owner, I want every result reproducible and every dependency proven and license-checked, so that I can defend every line of the project in an interview.

#### Acceptance Criteria

1. WHERE a number is reported in the README, in `docs/`, or in the Results_Table, THE System SHALL produce that number from a script in this repository and SHALL record, in the same document and within the same section as the number, the generating command, the seed, and the full 40-character commit hash.
2. WHERE a System component uses randomness, THE System SHALL expose a configurable seed through a documented Configuration parameter.
3. WHEN two runs of a System component execute with the same seed, the same Configuration, and the same model, THE System SHALL produce numeric results that are equal to within an absolute tolerance of 1e-6 for floating-point values and exactly equal for integer and categorical values, excluding timestamps, absolute file paths, and values the model provider documents as nondeterministic.
4. IF a run fails or misses a stated target, THEN THE System SHALL report, in the run output, the actual result and the identified cause, SHALL terminate with a non-zero exit status, and SHALL leave the stated target and the test unchanged.
5. THE System SHALL record, in the ENVIRONMENT_Doc, the exact version string of every tool, model, API, and dependency it relies on, including the Local_Model name and version and, where used, the Hosted_Model name and version.
6. WHEN a dependency, tool, or model version is recorded in the ENVIRONMENT_Doc, THE System SHALL include, alongside that version, at least one of the following verifiable artifacts obtained on the target machine: a package-manager query result showing that exact version, a documentation reference for that exact version, or an automated test that exercises the dependency and passes against that exact version.
7. WHERE a third-party package, model, or asset is added to the repository, THE System SHALL record, in the THIRD_PARTY_Doc, its name, its version, and its license identifier.
8. WHERE Corpus content, Evaluation_Set content, or a reused module is derived from an external source, THE System SHALL record the source and its license in the THIRD_PARTY_Doc and SHALL attribute reused Rag_Lite modules to their origin.
9. THE System SHALL read every secret, including any Hosted_Model API key, from an environment variable and SHALL list every secret-bearing file in `.gitignore` so that no such file is tracked in version control.
10. THE System SHALL pin the exact version of every dependency it declares.
11. WHERE a module contains more than one public function or exceeds 50 source lines, THE System SHALL provide unit tests achieving at least 80 percent statement coverage for that module, and THE README SHALL state the exact single command that runs those tests.
12. THE System SHALL remain a private repository without a public license, published package, or public release until the repository owner directs otherwise.
    - **Owner decision, 2026-10-21:** the repository was released publicly under the MIT License. This requirement is superseded.
13. IF a required secret environment variable is unset or empty when a System component that depends on it starts, THEN THE System SHALL terminate before issuing any request that requires that secret, SHALL emit an error indicating which environment variable is missing, and SHALL exit with a non-zero exit status.

### Requirement 1: Environment and local model server smoke test (M0)

**User Story:** As the repository owner, I want a validated environment and local model server, so that the open-weight model runs on the GPU before any pipeline work begins.

#### Acceptance Criteria

1. WHEN the smoke test runs, THE System SHALL record in the ENVIRONMENT_Doc the output of `nvidia-smi`, including the reported total GPU memory in mebibytes of the NVIDIA RTX 3080.
2. WHEN the smoke test runs, THE System SHALL start the Model_Server, send exactly one question to the OpenAI-compatible endpoint, and record in the ENVIRONMENT_Doc a generated Answer text containing at least one non-whitespace character.
3. WHILE the Local_Model generates the smoke-test Answer, THE System SHALL measure the peak GPU memory in mebibytes used between the start and completion of the generation request and SHALL record that value in the ENVIRONMENT_Doc.
4. THE System SHALL select a Local_Model whose recorded peak GPU memory during the smoke test does not exceed the total GPU memory reported by `nvidia-smi`.
5. THE System SHALL record in the ENVIRONMENT_Doc the Local_Model name, version, quantization, and license, and the Model_Server name and version.
6. IF the smoke test does not record a valid Answer within 120 seconds of sending the question, THEN THE System SHALL stop the attempt, report the failure and its identified cause, and SHALL record in the ENVIRONMENT_Doc the condition that failed.
7. WHERE a Hosted_Model is used, THE System SHALL record the Spend_Cap in United States dollars per month in the ENVIRONMENT_Doc and SHALL read the Hosted_Model API key from an environment variable.
8. WHEN the smoke test runs, THE System SHALL verify that the installed Python version is 3.11 or newer and that Docker is available, and IF either prerequisite is not met, THEN THE System SHALL report the failure and record in the ENVIRONMENT_Doc the prerequisite that failed.

### Requirement 2: Corpus and evaluation set (M1)

**User Story:** As the repository owner, I want a license-clear corpus and a graded evaluation set, so that answer quality can be measured against gold data.

#### Acceptance Criteria

1. THE System SHALL assemble a Corpus from a public, license-clear document collection and SHALL record the Corpus source and license in the THIRD_PARTY_Doc.
2. WHEN the Corpus is chunked, THE System SHALL assign each Passage a unique Passage_ID that is identical for the same Passage across repeated runs on the same Corpus, and IF two Passages are assigned the same Passage_ID, THEN THE System SHALL report every duplicated Passage_ID and the count of affected Passages.
3. THE Evaluation_Set SHALL contain at least 150 questions, and each question SHALL carry at least one gold supporting Passage_ID except for questions designated as Unanswerable_Questions.
4. THE Evaluation_Set SHALL contain at least 20 Unanswerable_Questions, each labeled as unanswerable and carrying no gold supporting Passage_ID.
5. WHERE an Evaluation_Set question is derived from a public QA dataset, THE System SHALL record that dataset and its license in the THIRD_PARTY_Doc.
6. WHEN an Evaluation_Set question specifies a gold supporting Passage_ID, THE System SHALL verify that the Passage_ID exists in the chunked Corpus, and IF any gold Passage_ID does not resolve to an existing Passage, THEN THE System SHALL report each unresolved gold Passage_ID and the total count of unresolved gold Passage_IDs.

### Requirement 3: Retrieval layer reused from Rag_Lite (M1)

**User Story:** As the repository owner, I want the retrieval baseline reproduced from Rag_Lite, so that retrieval quality is a known, measured starting point.

#### Acceptance Criteria

1. THE Retrieval_Layer SHALL use Rag_Lite for TF-IDF, BM25, dense embedding, hybrid fusion, and cross-encoder reranking rather than reimplementing those capabilities.
2. WHEN the Retrieval_Layer receives a query, THE Retrieval_Layer SHALL return a ranked list of Passages ordered by descending relevance score, each identified by its Passage_ID, with no duplicate Passage_ID in the list.
3. WHEN the retrieval baseline is evaluated on the Evaluation_Set, THE System SHALL report Recall@K for K equal to 5, 10, and 20 and nDCG at the same K values as numeric Measured_Values, each in the range 0.0 to 1.0.
4. IF the Retrieval_Layer receives a query that matches no Passage above its relevance scoring floor, THEN THE Retrieval_Layer SHALL return an empty ranked list rather than an error.
5. THE System SHALL record in the THIRD_PARTY_Doc how Rag_Lite is reused, stating whether it is installed as a pinned-version package (including the exact pinned version) or copied as named modules with attribution (including each copied module name).
6. WHERE Rag_Lite modules are copied into the repository, THE System SHALL attribute each copied module to Rag_Lite and SHALL record Rag_Lite's license in the THIRD_PARTY_Doc.

### Requirement 4: Answering pipeline with citations (M2)

**User Story:** As a user, I want answers that are grounded in retrieved passages and cite them, so that I can trust and verify each answer.

#### Acceptance Criteria

1. WHEN the Answering_Pipeline receives a query, THE Answering_Pipeline SHALL invoke the Retrieval_Layer with hybrid retrieval and cross-encoder reranking and SHALL build a prompt containing the reranked Passages.
2. WHEN the Answering_Pipeline builds a prompt, THE Answering_Pipeline SHALL instruct the Local_Model to answer only from the provided Passages and to cite the Passage_ID of each Passage used.
3. WHEN the Local_Model produces an Answer, THE Answering_Pipeline SHALL return the answer text together with the list of cited Passage_IDs, where every cited Passage_ID is drawn from the Passages included in the prompt.
4. IF the retrieval confidence for a query is below the configured confidence threshold, THEN THE Answering_Pipeline SHALL return a Refusal with the text "not found in the documents" and SHALL NOT invoke the Local_Model to produce substantive answer text.
5. WHEN an Unanswerable_Question is submitted to the Answering_Pipeline, THE Answering_Pipeline SHALL return a Refusal with the text "not found in the documents".
6. THE confidence threshold that triggers a Refusal SHALL be a documented configuration parameter expressed as a single numeric value in the range 0.0 to 1.0.

### Requirement 5: Citation validation (M2)

**User Story:** As a user, I want citations verified against the retrieved passages, so that answers cannot cite material the system did not retrieve.

#### Acceptance Criteria

1. WHEN the Answering_Pipeline produces an Answer, THE Citation_Validation SHALL compare every cited Passage_ID against the set of Passage_IDs retrieved for that query.
2. IF an Answer cites a Passage_ID that is not in the retrieved set for that query, THEN THE Citation_Validation SHALL reject the Answer and SHALL report every offending Passage_ID.
3. WHEN the Citation_Validation rejects an Answer, THE System SHALL return a response that contains no uncited claims, by either returning a corrected Answer whose cited Passage_IDs are all in the retrieved set or returning a Refusal with the text "not found in the documents".
4. IF an Answer contains a factual claim that carries no cited Passage_ID, THEN THE Citation_Validation SHALL reject the Answer.
5. THE System SHALL provide unit tests covering Citation_Validation acceptance of an Answer whose cited Passage_IDs are all in the retrieved set and rejection of an Answer citing at least one Passage_ID outside the retrieved set.

### Requirement 6: Guardrails against injected instructions (M4)

**User Story:** As the repository owner, I want retrieved text treated as data, so that instructions embedded in documents do not control the model, and I want the residual risk measured.

#### Acceptance Criteria

1. WHEN the Answering_Pipeline builds a prompt, THE Answering_Pipeline SHALL delimit the retrieved Passage text from the instructions and SHALL instruct the Local_Model to treat Passage text as data and to ignore any instructions contained within Passages.
2. WHEN the Injection_Test runs, THE Injection_Test SHALL run the Answering_Pipeline over a set of Passages that contain embedded instructions and SHALL report, as a Measured_Value in the range 0.0 to 1.0, the fraction of those cases in which the Local_Model followed an embedded instruction.
3. WHEN the Injection_Test classifies a case, THE Injection_Test SHALL count the case as "followed" only if the Answer contains the observable effect that the embedded instruction directed, according to a documented classification rule.
4. WHEN the Injection_Test reports its result, THE System SHALL record the Measured_Value with its generating command, its integer seed, and its full 40-character commit hash.
5. THE Injection_Test SHALL use at least 20 distinct injected-instruction cases, where two cases are distinct if their embedded instructions differ in text.
6. IF the Injection_Test cannot classify a case as followed or not followed, THEN THE Injection_Test SHALL report that case as unclassified and SHALL exclude it from the followed-fraction denominator while reporting the count of unclassified cases.

### Requirement 7: Evaluation harness (M3)

**User Story:** As the repository owner, I want an automated evaluation harness, so that retrieval and answer quality are measured consistently with confidence intervals.

#### Acceptance Criteria

1. WHEN the evaluation harness runs on the Evaluation_Set, THE System SHALL report Recall@K and nDCG for retrieval as numeric Measured_Values, each in the range 0.0 to 1.0.
2. WHEN an Evaluation_Set question carries a gold answer, THE System SHALL report exact-match and token-level F1 of the Answer against that gold answer as numeric Measured_Values, each in the range 0.0 to 1.0.
3. WHEN the evaluation harness runs, THE System SHALL report citation precision and citation recall of the Answer's cited Passage_IDs against the gold supporting Passage_IDs as numeric Measured_Values, each in the range 0.0 to 1.0.
4. WHEN the evaluation harness runs on the Unanswerable_Questions, THE System SHALL report refusal accuracy as the fraction, in the range 0.0 to 1.0, of Unanswerable_Questions for which the System returned a Refusal.
5. WHEN the LLM_Judge scores Faithfulness, THE System SHALL report, as a numeric Measured_Value in the range -1.0 to 1.0, the Cohen's kappa agreement between the LLM_Judge labels and the repository owner's labels over at least 50 samples.
6. WHEN the evaluation harness reports a metric in the Results_Table, THE System SHALL report a 95 percent bootstrap confidence interval computed from at least 1,000 bootstrap resamples for that metric.
7. THE evaluation harness SHALL be invocable from a single non-interactive command.
8. WHEN the evaluation harness runs twice on identical inputs with the same seed and the same model, THE System SHALL produce metric values that are equal to within an absolute tolerance of 1e-6, excluding values the model provider documents as nondeterministic.
9. IF an Evaluation_Set input required by a metric is missing or malformed, THEN THE evaluation harness SHALL report the affected metric as not computed, SHALL report the identified cause, and SHALL continue computing the remaining metrics.

### Requirement 8: Configuration experiments (M4)

**User Story:** As the repository owner, I want measured comparisons across configurations, so that I can show quality, latency, and cost trade-offs.

#### Acceptance Criteria

1. THE System SHALL evaluate at least four Configurations: BM25-only; hybrid; hybrid plus reranking; and hybrid plus reranking run with both a smaller and a larger model.
2. WHEN an Experiment runs, THE System SHALL report, for each Configuration, answer-quality metrics, retrieval metrics, P50 latency in milliseconds, P95 latency in milliseconds, and cost per 1,000 questions in United States dollars reported to at least two decimal places.
3. WHEN an Experiment completes, THE System SHALL generate the Results_Table as a file produced by a script, and no reported number in the Results_Table SHALL be modified by any means other than regenerating the file with the script.
4. THE Results_Table SHALL include a 95 percent bootstrap confidence interval computed from at least 1,000 bootstrap resamples for every reported quality and retrieval metric.
5. WHERE a Configuration uses the Hosted_Model, THE System SHALL compute its reported cost per 1,000 questions from the Hosted_Model's recorded per-token prices and the recorded token counts for that Configuration.
6. IF a Configuration fails to complete its Experiment run, THEN THE System SHALL report that Configuration as failed with its identified cause and SHALL still generate the Results_Table for the Configurations that completed.

### Requirement 9: Service, logging, and orchestration (M5)

**User Story:** As a user, I want a running service with streaming answers and observability, so that I can query the system and monitor it.

#### Acceptance Criteria

1. WHEN the FastAPI endpoint receives a request carrying a well-formed query, THE System SHALL return an Answer containing the answer text and the list of its cited Passage_IDs.
2. IF the FastAPI endpoint receives a request whose query is empty, is not valid text, or exceeds the documented maximum query length, THEN THE System SHALL reject the request without invoking the Answering_Pipeline and SHALL return a response indicating the query was invalid, and SHALL leave service state unchanged.
3. WHEN a client requests streaming, THE FastAPI endpoint SHALL emit the first chunk of Answer text within the documented first-chunk timeout and SHALL continue emitting the remaining Answer text in successive chunks until the Answer is complete.
4. WHEN the FastAPI endpoint receives a request, THE System SHALL assign that request a unique request identifier and SHALL include that request identifier in every structured log entry emitted for that request.
5. WHEN the System processes a request, THE System SHALL emit at least one structured log entry per request that records the request identifier, the request start time, the final outcome of the request, and the request duration in milliseconds.
6. WHEN a client reads the metrics endpoint, THE System SHALL return the current request-count and request-latency metrics in a machine-readable response.
7. WHEN the documented single Docker Compose command is run, THE System SHALL start the service together with every dependency it requires and SHALL reach a state in which the FastAPI endpoint accepts requests.
8. THE System SHALL read every secret used by the service, including any Hosted_Model API key, from an environment variable rather than from a tracked file.
9. IF a required secret environment variable is absent when the service starts, THEN THE System SHALL fail to start and SHALL report, in the service start-up output, which required secret was missing.

### Requirement 10: Continuous-integration regression gate (M5)

**User Story:** As the repository owner, I want a CI gate on quality, so that no change ships if faithfulness or citation precision regresses.

#### Acceptance Criteria

1. WHEN a change is pushed to the repository's default branch or a pull request targeting the default branch is opened or updated, THE CI_Pipeline SHALL run the unit tests and evaluate an evaluation subset of the Evaluation_Set that contains at least the documented minimum number of questions.
2. WHEN the evaluation subset has been evaluated, THE CI_Pipeline SHALL compare the evaluation-subset Faithfulness and citation precision against the corresponding values recorded in the Baseline_File.
3. IF the evaluation-subset Faithfulness or citation precision is lower than its Baseline_File value by more than the documented regression threshold, expressed as an absolute drop in that metric's native units, THEN THE CI_Pipeline SHALL report a failed status for that run.
4. IF any unit test fails during a CI run, THEN THE CI_Pipeline SHALL report a failed status for that run.
5. IF the Baseline_File is missing or cannot be read during a CI run, THEN THE CI_Pipeline SHALL report a failed status for that run and SHALL report, in the run output, that the Baseline_File was missing or unreadable.
6. THE documented regression threshold SHALL be a configuration value recorded in the repository, and THE Baseline_File SHALL record the baseline Measured_Values with their generating command, seed, and full 40-character commit hash.

### Requirement 11: README and reproducibility (M5)

**User Story:** As a reader, I want a README with measured results, so that I can understand, run, and reproduce the project.

#### Acceptance Criteria

1. THE README SHALL contain an architecture diagram, run instructions, a Results_Table that includes a confidence interval for every reported metric, a cost-and-latency trade-off section, a known-limitations section, and a demo GIF.
2. THE README SHALL state, for each test suite, the exact single command that runs that suite, and SHALL state, for each reported result, the exact single command that reproduces that result.
3. THE README SHALL state the exact single Docker Compose command that starts the service, and that command SHALL match the single Docker Compose command required by Requirement 9.
4. WHERE a number appears in the README, THE System SHALL source that number from the Results_Table or another file generated by a script in this repository rather than from a hand-entered value.
5. IF a number in the README cannot be traced to the Results_Table or another generated file, THEN THE System SHALL report that number as untraceable in the check output and SHALL report a failed status for that check.

## Out of Scope

The following are explicitly out of scope: fine-tuning of any model; user accounts and authentication; a front end beyond a minimal demo page; and paid vector databases. The optional Spring Boot gateway in front of the FastAPI service is permitted but not required.
