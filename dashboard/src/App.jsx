
import { useCallback, useEffect, useState } from "react";
import "./App.css";

const API = "http://127.0.0.1:8000";

// -------------------- API REQUEST HELPER --------------------

async function request(path, options = {}, token = "") {
  const response = await fetch(`${API}${path}`, {
    ...options,
    headers: {
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...options.headers,
    },
  });

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : `Request failed (${response.status})`
    );
  }

  return data;
}

// -------------------- UTILITY FUNCTIONS --------------------

function array(value) {
  if (Array.isArray(value)) return value;
  if (value == null || value === "") return [];
  return [value];
}

function display(value) {
  if (value == null || value === "") return "Not available";
  if (typeof value === "object") {
    return JSON.stringify(value, null, 2);
  }
  return String(value);
}

function confidence(value) {
  if (value == null || value === "") return "Not available";

  const number = Number(value);

  if (!Number.isFinite(number)) return String(value);

  return `${Math.round(number <= 1 ? number * 100 : number)}%`;
}

function getRows(data) {
  if (Array.isArray(data)) return data;
  if (Array.isArray(data?.items)) return data.items;
  if (Array.isArray(data?.results)) return data.results;
  return [];
}

// -------------------- MAIN APPLICATION --------------------

function App() {
  const [token, setToken] = useState("");
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("");
  const [loginLoading, setLoginLoading] = useState(false);

  const [overview, setOverview] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [investigations, setInvestigations] = useState([]);
  const [responseActions, setResponseActions] = useState([]);
  const [auditLogs, setAuditLogs] = useState([]);
  const [selectedAlertId, setSelectedAlertId] = useState(null);
  const [alertDetails, setAlertDetails] = useState(null);
  const [investigation, setInvestigation] = useState(null);

  const [loading, setLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [savingWorkflow, setSavingWorkflow] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [workflowChoice, setWorkflowChoice] = useState("NEW");

  // -------------------- LOGIN --------------------

  async function login(event) {
    event.preventDefault();

    setLoginLoading(true);
    setError("");
    setMessage("");

    try {
      const data = await request("/auth/login", {
        method: "POST",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body: new URLSearchParams({ username, password }),
      });

      if (!data.access_token) {
        throw new Error("Login response did not contain an access token.");
      }

      setToken(data.access_token);
      setPassword("");
      setMessage("Signed in successfully.");
    } catch (err) {
      setError(`Login failed: ${err.message}`);
    } finally {
      setLoginLoading(false);
    }
  }

  // -------------------- LOAD DASHBOARD DATA --------------------

  const loadOverview = useCallback(async () => {
    if (!token) return;

    setLoading(true);
    setError("");

    try {
      const [
        overviewData,
        incidentData,
        investigationData,
        responseActionData,
        auditLogData,
      ] = await Promise.all([
        request("/api/security/overview", {}, token),
        request("/incidents", {}, token),
        request("/investigations", {}, token),
        request("/response-actions", {}, token),
        request("/audit-logs", {}, token).catch(() => []),
      ]);

      const incidentRows = getRows(incidentData);
      const investigationRows = getRows(investigationData);
      const responseActionRows = getRows(responseActionData);
      const auditLogRows = getRows(auditLogData);

      setOverview(overviewData);
      setInvestigations(investigationRows);
      setResponseActions(responseActionRows);
      setAuditLogs(auditLogRows);

      const alertRows = incidentRows.map((incident) => {
        const linkedInvestigation = investigationRows.find(
          (item) => Number(item.incident_id) === Number(incident.id)
        );

        return {
          ...incident,
          alert_id: incident.id,
          detection_confidence: incident.confidence,
          investigation_id: linkedInvestigation?.id ?? null,
          processing_status:
            linkedInvestigation?.processing_status ?? "NOT_STARTED",
          workflow_status:
            linkedInvestigation?.workflow_status ?? "NEW",
        };
      });

      setAlerts(alertRows);

      if (
        selectedAlertId != null &&
        !alertRows.some(
          (item) => Number(item.alert_id) === Number(selectedAlertId)
        )
      ) {
        setSelectedAlertId(null);
        setAlertDetails(null);
        setInvestigation(null);
      } else if (selectedAlertId != null) {
        const refreshedAlert = alertRows.find(
          (item) => Number(item.alert_id) === Number(selectedAlertId)
        );

        if (refreshedAlert) {
          const linkedInvestigation = investigationRows.find(
            (item) =>
              Number(item.incident_id) ===
              Number(refreshedAlert.alert_id)
          );

          if (
            investigation &&
            linkedInvestigation &&
            Number(investigation.id) === Number(linkedInvestigation.id)
          ) {
            setInvestigation((previous) => ({
              ...previous,
              processing_status:
                linkedInvestigation.processing_status,
              workflow_status: linkedInvestigation.workflow_status,
            }));

            setWorkflowChoice(
              linkedInvestigation.workflow_status ?? "NEW"
            );
          }
        }
      }
    } catch (err) {
      setError(`Could not load security data: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, [token, selectedAlertId, investigation]);

  // -------------------- SELECT ALERT --------------------

  async function selectAlert(selectedAlert) {
    setSelectedAlertId(selectedAlert.alert_id);
    setAlertDetails(null);
    setInvestigation(null);
    setWorkflowChoice("NEW");
    setDetailLoading(true);
    setError("");
    setMessage("");

    try {
      // Fetch the latest incident details.
      const detail = await request(
        `/incidents/${encodeURIComponent(selectedAlert.alert_id)}`,
        {},
        token
      );

      // Find the investigation linked to this incident.
      const linkedInvestigation = investigations.find(
        (item) => Number(item.incident_id) === Number(detail.id)
      );

      setAlertDetails({
        ...detail,
        alert_id: detail.id,
        detection_confidence: detail.confidence,
        investigation_id: linkedInvestigation?.id ?? null,
        label: detail.label ?? detail.attack_type,
        dataset: detail.dataset ?? null,
      });

      if (linkedInvestigation) {
        // IMPORTANT:
        // Fetch the latest investigation from the backend.
        // Do not depend on potentially stale list data.
        const latestInvestigation = await request(
          `/investigations/${encodeURIComponent(linkedInvestigation.id)}`,
          {},
          token
        );

        const reportData = latestInvestigation.report_data ?? {};

        setInvestigation({
          ...reportData,

          // Keep database metadata available to the dashboard.
          id: latestInvestigation.id,
          investigation_id: latestInvestigation.id,
          alert_id: latestInvestigation.incident_id,
          report_data: reportData,

          // Prefer the report's generation timestamp.
          created_at:
            reportData.generated_at ??
            latestInvestigation.updated_at ??
            latestInvestigation.created_at,

          processing_status: latestInvestigation.processing_status,
          workflow_status: latestInvestigation.workflow_status,
        });

        setWorkflowChoice(
          latestInvestigation.workflow_status ?? "NEW"
        );
      }
    } catch (err) {
      setError(`Could not load alert details: ${err.message}`);
    } finally {
      setDetailLoading(false);
    }
  }

  // -------------------- SAVE WORKFLOW --------------------

  async function saveWorkflow() {
    if (investigation?.id == null || !workflowChoice) return;

    setSavingWorkflow(true);
    setError("");
    setMessage("");

    try {
      const updated = await request(
        `/investigations/${encodeURIComponent(investigation.id)}/workflow`,
        {
          method: "PATCH",
          body: JSON.stringify({
            workflow_status: workflowChoice,
          }),
        },
        token
      );

      setInvestigation((previous) => ({
        ...previous,
        ...updated,

        // Keep the report fields and metadata together.
        ...((updated.report_data ?? previous.report_data)
          ? updated.report_data ?? previous.report_data
          : {}),

        id: updated.id ?? previous.id,
        investigation_id:
          updated.id ?? previous.investigation_id,
        workflow_status: updated.workflow_status ?? workflowChoice,
      }));

      setMessage("Investigation workflow status updated.");

      await loadOverview();
    } catch (err) {
      setError(`Could not update workflow: ${err.message}`);
    } finally {
      setSavingWorkflow(false);
    }
  }

  // -------------------- RESPONSE ACTION REVIEW --------------------

  async function reviewResponseAction(actionId, decision) {
    const notes = window.prompt(
      `Enter review notes for ${decision.toLowerCase()} (optional):`,
      ""
    );

    // Cancel means the reviewer does not want to submit the decision.
    if (notes === null) return;

    setError("");
    setMessage("");

    try {
      await request(
        `/response-actions/${encodeURIComponent(actionId)}/approval`,
        {
          method: "POST",
          body: JSON.stringify({
            decision,
            notes,
          }),
        },
        token
      );

      setMessage(`Response action #${actionId} was ${decision.toLowerCase()}.`);
      await loadOverview();
    } catch (err) {
      setError(`Could not review response action: ${err.message}`);
    }
  }

  // -------------------- LOGOUT --------------------

  function logout() {
    setToken("");
    setOverview(null);
    setAlerts([]);
    setInvestigations([]);
    setResponseActions([]);
    setAuditLogs([]);
    setSelectedAlertId(null);
    setAlertDetails(null);
    setInvestigation(null);
    setPassword("");
    setWorkflowChoice("NEW");
    setError("");
    setMessage("");
  }

  // -------------------- INITIAL DATA LOAD --------------------

  useEffect(() => {
    if (token) {
      loadOverview();
    }
  }, [token, loadOverview]);

  // -------------------- LOGIN SCREEN --------------------

  if (!token) {
    return (
      <main className="soc">
        <section className="panel login-panel">
          <p className="eyebrow">
            THREAT INTELLIGENCE &amp; INVESTIGATION
          </p>

          <h1>AI-NIDS Security Center</h1>

          <p className="muted">
            Sign in to review security alerts and investigations.
          </p>

          {error && <p className="error">{error}</p>}
          {message && <p className="success-message">{message}</p>}

          <form onSubmit={login} className="login-form">
            <label htmlFor="username">Username</label>

            <input
              id="username"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              autoComplete="username"
              required
            />

            <label htmlFor="password">Password</label>

            <input
              id="password"
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              autoComplete="current-password"
              required
            />

            <button type="submit" disabled={loginLoading}>
              {loginLoading ? "Signing in..." : "Sign in"}
            </button>
          </form>
        </section>
      </main>
    );
  }

  // -------------------- REPORT DATA --------------------

  const overviewData = overview ?? {};
  const alert = alertDetails ?? {};
  const report = investigation ?? {};

  const iocs = array(report.iocs);
  const techniques = array(report.mitre_technique);
  const evidence = array(report.evidence);
  const sources = array(report.sources);
  const uncertainties = array(report.uncertainties);
  const recommendations = array(report.recommendations);

  // These field names match investigation_engine.py.
  const facts = array(report.observed_facts);
  const explanations = array(report.possible_explanations);

  // -------------------- DASHBOARD --------------------

  return (
    <main className="soc">
      <header className="topbar">
        <div>
          <p className="eyebrow">SECURITY OPERATIONS CENTER</p>

          <h1>AI-NIDS Investigation Console</h1>

          <p className="muted">
            Detection · Threat intelligence · AI investigation · Review
          </p>
        </div>

        <div className="toolbar">
          <button onClick={loadOverview} disabled={loading}>
            {loading ? "Refreshing..." : "Refresh"}
          </button>

          <button className="secondary" onClick={logout}>
            Logout
          </button>
        </div>
      </header>

      {error && <p className="error">{error}</p>}
      {message && <p className="success-message">{message}</p>}

      {/* -------------------- METRICS -------------------- */}

      <section className="metrics">
        <Metric
          label="Total alerts"
          value={overviewData.total_alerts}
        />

        <Metric
          label="High / critical"
          value={overviewData.high_critical_alerts}
        />

        <Metric
          label="Investigations"
          value={overviewData.total_investigations}
        />

        <Metric
          label="Completed investigations"
          value={overviewData.completed_investigations}
        />

        <Metric
          label="Awaiting review"
          value={overviewData.awaiting_review}
        />
      </section>

      {/* -------------------- RESPONSE ACTION REVIEW -------------------- */}

      <section className="panel response-review-panel">
        <div className="section-heading">
          <div>
            <h2>Response Action Review</h2>
            <p className="muted">
              Review recommendations and record an administrator decision.
              No network action is executed by approving or rejecting here.
            </p>
          </div>
          <strong>{responseActions.length} response actions</strong>
        </div>

        {responseActions.length === 0 ? (
          <p className="muted">
            No response actions found. Create a response recommendation from
            the backend to see it here.
          </p>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Action ID</th>
                  <th>Incident</th>
                  <th>Recommendation</th>
                  <th>Priority</th>
                  <th>Approval</th>
                  <th>Execution</th>
                  <th>Notes</th>
                  <th>Review</th>
                </tr>
              </thead>
              <tbody>
                {responseActions.map((action) => (
                  <tr key={action.id}>
                    <td>#{action.id}</td>
                    <td>#{action.incident_id}</td>
                    <td>{display(action.recommended_action)}</td>
                    <td>{display(action.priority)}</td>
                    <td>
                      <span
                        className={`severity ${String(
                          action.approval_status ?? "unknown"
                        ).toLowerCase()}`}
                      >
                        {display(action.approval_status)}
                      </span>
                    </td>
                    <td>{display(action.execution_status)}</td>
                    <td>{display(action.notes)}</td>
                    <td>
                      {action.approval_status === "PENDING" ? (
                        <div className="toolbar">
                          <button
                            type="button"
                            onClick={() =>
                              reviewResponseAction(action.id, "APPROVED")
                            }
                          >
                            Approve
                          </button>
                          <button
                            type="button"
                            className="secondary"
                            onClick={() =>
                              reviewResponseAction(action.id, "REJECTED")
                            }
                          >
                            Reject
                          </button>
                        </div>
                      ) : (
                        <span className="muted">Reviewed</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>


      {/* -------------------- AUDIT HISTORY -------------------- */}

      <section className="panel audit-history-panel">
        <div className="section-heading">
          <div>
            <h2>Audit History</h2>
            <p className="muted">
              Administrator decisions recorded for response actions.
            </p>
          </div>
          <strong>{auditLogs.length} audit records</strong>
        </div>

        {auditLogs.length === 0 ? (
          <p className="muted">
            No audit records found yet. New administrator approvals and
            rejections will appear here.
          </p>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Audit ID</th>
                  <th>Action ID</th>
                  <th>Incident</th>
                  <th>Administrator</th>
                  <th>Decision</th>
                  <th>Notes</th>
                  <th>Reviewed At</th>
                </tr>
              </thead>
              <tbody>
                {auditLogs.map((log) => (
                  <tr key={log.id}>
                    <td>#{log.id}</td>
                    <td>#{log.response_action_id}</td>
                    <td>#{log.incident_id}</td>
                    <td>{display(log.admin_username)}</td>
                    <td>
                      <span
                        className={`severity ${String(
                          log.decision ?? "unknown"
                        ).toLowerCase()}`}
                      >
                        {display(log.decision)}
                      </span>
                    </td>
                    <td>{display(log.notes)}</td>
                    <td>
                      {log.reviewed_at
                        ? new Date(log.reviewed_at).toLocaleString()
                        : "Not available"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

            {/* -------------------- WORKSPACE -------------------- */}

      <section className="workspace">
        <aside className="panel alert-panel">
          <div className="section-heading">
            <div>
              <h2>Security alerts</h2>

              <p className="muted">
                {alerts.length} alerts loaded from the incidents API
              </p>
            </div>
          </div>

          {loading && alerts.length === 0 && (
            <p>Loading alerts...</p>
          )}

          {!loading && alerts.length === 0 && (
            <p className="muted">
              No alerts returned by the backend.
            </p>
          )}

          <div className="alert-list">
            {alerts.map((item) => (
              <button
                key={item.alert_id}
                className={`alert-row ${
                  Number(selectedAlertId) === Number(item.alert_id)
                    ? "selected"
                    : ""
                }`}
                onClick={() => selectAlert(item)}
              >
                <div className="row-between">
                  <strong>Alert #{item.alert_id}</strong>

                  <span
                    className={`severity ${String(
                      item.severity ?? "unknown"
                    ).toLowerCase()}`}
                  >
                    {display(item.severity)}
                  </span>
                </div>

                <span>{display(item.attack_type)}</span>

                <span className="muted">
                  {display(item.source_ip)} →{" "}
                  {display(item.destination_ip)}
                </span>

                <span className="muted">
                  Dataset: {display(item.dataset)} · Confidence:{" "}
                  {confidence(item.detection_confidence)}
                </span>

                <span className="muted">
                  Workflow: {display(item.workflow_status)} · Report:{" "}
                  {display(item.processing_status)}
                </span>
              </button>
            ))}
          </div>

          {overviewData.alerts_by_attack_type && (
            <section className="attack-summary">
              <h3>Alerts by attack type</h3>

              {Object.entries(
                overviewData.alerts_by_attack_type
              ).length === 0 ? (
                <p className="muted">
                  No attack categories available.
                </p>
              ) : (
                <ul>
                  {Object.entries(
                    overviewData.alerts_by_attack_type
                  ).map(([name, count]) => (
                    <li key={name}>
                      <span>{name}</span>
                      <strong>{count}</strong>
                    </li>
                  ))}
                </ul>
              )}
            </section>
          )}
        </aside>

        {/* -------------------- DETAIL COLUMN -------------------- */}

        <section className="detail-column">
          {!selectedAlertId && (
            <section className="panel empty-state">
              <h2>Investigation workspace</h2>

              <p className="muted">
                Select an alert to view its details and linked
                investigation report.
              </p>
            </section>
          )}

          {detailLoading && (
            <section className="panel">
              Loading alert and investigation details...
            </section>
          )}

          {selectedAlertId && !detailLoading && alertDetails && (
            <>
              {/* -------------------- ALERT DETAILS -------------------- */}

              <section className="panel">
                <h2>Alert details</h2>

                <div className="detail-grid">
                  <Field label="Alert ID" value={alert.alert_id} />

                  <Field
                    label="Investigation ID"
                    value={alert.investigation_id}
                  />

                  <Field label="Created at" value={alert.created_at} />

                  <Field label="Source IP" value={alert.source_ip} />

                  <Field
                    label="Destination IP"
                    value={alert.destination_ip}
                  />

                  <Field
                    label="Attack type"
                    value={alert.attack_type}
                  />

                  <Field label="Dataset" value={alert.dataset} />

                  <Field
                    label="Detection confidence"
                    value={confidence(alert.detection_confidence)}
                  />

                  <Field label="Detection label" value={alert.label} />

                  <Field label="Severity" value={alert.severity} />

                  <Field
                    label="Incident status"
                    value={alert.status}
                  />
                </div>
              </section>

              {/* -------------------- NO INVESTIGATION -------------------- */}

              {!investigation && (
                <section className="panel">
                  <h2>Investigation report</h2>

                  <p className="muted">
                    No investigation report is linked to this alert yet.
                  </p>
                </section>
              )}

              {/* -------------------- INVESTIGATION REPORT -------------------- */}

              {investigation && (
                <>
                  <section className="panel">
                    <h2>AI investigation summary</h2>

                    <p>{display(report.summary)}</p>

                    <div className="detail-grid">
                      <Field
                        label="Investigation ID"
                        value={report.investigation_id}
                      />

                      <Field
                        label="Report created"
                        value={report.created_at}
                      />

                      <Field
                        label="Report status"
                        value={report.processing_status}
                      />

                      <Field
                        label="Attack type"
                        value={alert.attack_type}
                      />

                      <Field
                        label="Label"
                        value={alert.label}
                      />

                      <Field
                        label="Severity"
                        value={alert.severity}
                      />

                      <Field
                        label="Dataset"
                        value={alert.dataset}
                      />

                      <Field
                        label="Confidence"
                        value={confidence(alert.detection_confidence)}
                      />

                      <Field
                        label="Analysis type"
                        value={report.analysis_type}
                      />
                    </div>

                    {report.analysis_type ===
                      "RULE_BASED_PRELIMINARY" && (
                      <p className="muted">
                        This report uses rule-based preliminary analysis.
                        It is not an AI-generated or independently
                        verified threat assessment.
                      </p>
                    )}
                  </section>

                  {/* -------------------- IOC EXTRACTION -------------------- */}

                  <section className="panel">
                    <h2>IOC extraction &amp; enrichment</h2>

                    <p className="muted">
                      Extracted indicators and threat-intelligence context
                    </p>

                    <div className="table-wrap">
                      <table>
                        <thead>
                          <tr>
                            <th>Type</th>
                            <th>Value</th>
                            <th>Enrichment</th>
                          </tr>
                        </thead>

                        <tbody>
                          {iocs.map((ioc, index) => (
                            <tr
                              key={`${ioc?.value ?? "ioc"}-${index}`}
                            >
                              <td>{display(ioc?.type)}</td>
                              <td>{display(ioc?.value)}</td>

                              <td>
                                <pre>
                                  {display(
                                    ioc?.enrichment ??
                                      ioc?.threat_intelligence
                                  )}
                                </pre>
                              </td>
                            </tr>
                          ))}

                          {iocs.length === 0 && (
                            <tr>
                              <td colSpan="3">
                                No IOC data returned.
                              </td>
                            </tr>
                          )}
                        </tbody>
                      </table>
                    </div>
                  </section>

                  {/* -------------------- MITRE ATT&CK -------------------- */}

                  <section className="panel">
                    <h2>MITRE ATT&amp;CK mapping</h2>

                    <div className="table-wrap">
                      <table>
                        <thead>
                          <tr>
                            <th>Technique ID</th>
                            <th>Technique</th>
                            <th>Candidate status</th>
                          </tr>
                        </thead>

                        <tbody>
                          {techniques.map((technique, index) => (
                            <tr
                              key={`${
                                technique?.technique_id ?? "technique"
                              }-${index}`}
                            >
                              <td>
                                {display(technique?.technique_id)}
                              </td>

                              <td>{display(technique?.name)}</td>

                              <td>
                                {display(technique?.candidate_status)}
                              </td>
                            </tr>
                          ))}

                          {techniques.length === 0 && (
                            <tr>
                              <td colSpan="3">
                                No technique mappings returned.
                              </td>
                            </tr>
                          )}
                        </tbody>
                      </table>
                    </div>
                  </section>

                  {/* -------------------- INVESTIGATION REASONING -------------------- */}

                  <section className="panel">
                    <h2>Investigation reasoning</h2>

                    <TextList
                      title="Observed facts"
                      items={facts}
                    />

                    <TextList
                      title="Possible explanations"
                      items={explanations}
                    />

                    <TextList
                      title="Uncertainties and limitations"
                      items={uncertainties}
                    />
                  </section>

                  {/* -------------------- EVIDENCE -------------------- */}

                  <section className="panel">
                    <h2>Investigation evidence</h2>

                    <p className="muted">
                      Evidence and context returned by the investigation
                      engine. This section does not imply that RAG retrieval
                      has been implemented.
                    </p>

                    {evidence.length === 0 && (
                      <p className="muted">
                        No evidence returned.
                      </p>
                    )}

                    {evidence.map((item, index) => (
                      <article
                        className="evidence-card"
                        key={index}
                      >
                        <div className="row-between">
                          <strong>
                            {display(
                              item?.document_source ??
                                item?.source ??
                                "Evidence source"
                            )}
                          </strong>

                          <span className="muted">
                            {item?.similarity_score == null
                              ? ""
                              : `Similarity: ${Number(
                                  item.similarity_score
                                ).toFixed(3)}`}
                          </span>
                        </div>

                        <p>
                          {typeof item?.detail === "object" &&
                          item?.detail !== null
                            ? JSON.stringify(item.detail, null, 2)
                            : display(
                                item?.retrieved_text ?? item?.detail
                              )}
                        </p>
                      </article>
                    ))}

                    <TextList
                      title="Report sources"
                      items={sources}
                    />
                  </section>

                  {/* -------------------- RECOMMENDATIONS -------------------- */}

                  <section className="panel">
                    <h2>Defensive recommendations</h2>

                    <TextList
                      title="Recommended actions"
                      items={recommendations}
                    />

                    <p className="muted">
                      Recommendations require human review. Displaying
                      them does not execute a network action.
                    </p>
                  </section>

                  {/* -------------------- WORKFLOW -------------------- */}

                  <section className="panel">
                    <h2>Investigation workflow</h2>

                    <div className="detail-grid">
                      <Field
                        label="Processing status"
                        value={report.processing_status}
                      />

                      <Field
                        label="Human workflow status"
                        value={report.workflow_status}
                      />
                    </div>

                    <label htmlFor="workflow-status">
                      Update workflow status
                    </label>

                    <div className="toolbar workflow-controls">
                      <select
                        id="workflow-status"
                        value={workflowChoice}
                        onChange={(event) =>
                          setWorkflowChoice(event.target.value)
                        }
                      >
                        <option value="NEW">New</option>
                        <option value="IN_REVIEW">In review</option>
                        <option value="REVIEWED">Reviewed</option>
                        <option value="RESOLVED">Resolved</option>
                      </select>

                      <button
                        onClick={saveWorkflow}
                        disabled={
                          savingWorkflow ||
                          !workflowChoice ||
                          investigation.id == null
                        }
                      >
                        {savingWorkflow ? "Saving..." : "Save status"}
                      </button>
                    </div>
                  </section>

                  {/* -------------------- RAW REPORT -------------------- */}

                  <section className="panel">
                    <h2>Report details</h2>

                    <p className="muted">
                      Investigation report data returned by the backend
                    </p>

                    <pre className="report-json">
                      {JSON.stringify(report, null, 2)}
                    </pre>
                  </section>
                </>
              )}
            </>
          )}
        </section>
      </section>

      <footer>
        AI-NIDS Response System · Investigation findings are
        evidence-based assessments, not guaranteed conclusions.
      </footer>
    </main>
  );
}

// -------------------- REUSABLE COMPONENTS --------------------

function Metric({ label, value }) {
  return (
    <article className="metric-card">
      <span className="muted">{label}</span>
      <strong>{value == null ? "—" : value}</strong>
    </article>
  );
}

function Field({ label, value }) {
  return (
    <div className="field">
      <span className="muted">{label}</span>
      <strong>{display(value)}</strong>
    </div>
  );
}

function TextList({ title, items }) {
  return (
    <div className="text-list">
      <h3>{title}</h3>

      {items.length === 0 ? (
        <p className="muted">No data returned.</p>
      ) : (
        <ul>
          {items.map((item, index) => (
            <li key={index}>
              {typeof item === "object" && item !== null ? (
                <pre>{JSON.stringify(item, null, 2)}</pre>
              ) : (
                display(item)
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default App;
