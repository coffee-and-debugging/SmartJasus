const checkButton = document.getElementById("checkEmail");
const statusEl = document.getElementById("status");
const resultListEl = document.getElementById("resultList");

checkButton.addEventListener("click", () => {
    checkLatestInboxEmails();
});

// Load results immediately when popup opens
document.addEventListener("DOMContentLoaded", () => {
    checkLatestInboxEmails();
});

function getBackendBaseUrl() {
    return new Promise((resolve) => {
        chrome.storage.local.get(["backendBaseUrl"], (data) => {
            resolve((data.backendBaseUrl || "http://127.0.0.1:5000").replace(/\/$/, ""));
        });
    });
}

function setStatus(message) {
    if (!statusEl) return;
    statusEl.innerText = `Status: ${message}`;
}

function clearResults() {
    if (!resultListEl) return;
    resultListEl.innerHTML = "";
}

function renderResults(items) {
    clearResults();

    if (!items.length) {
        const empty = document.createElement("div");
        empty.className = "result-item";
        empty.innerText = "No inbox emails available to display.";
        if (resultListEl) {
            resultListEl.appendChild(empty);
        }
        return;
    }

    items.forEach((item) => {
        const card = document.createElement("div");
        card.className = "result-item";

        const subject = document.createElement("div");
        subject.className = "result-subject";
        subject.innerText = item.subject || "(No Subject)";

        const prediction = (item.prediction || "unknown").toLowerCase();
        const confidenceValue = Number(item.confidence || item.probability || 0);
        const confidencePct = `${Math.round(confidenceValue * 100)}%`;

        const meta = document.createElement("div");
        meta.className = "result-meta";

        const badge = document.createElement("span");
        badge.className = `badge ${prediction === "phishing" ? "phishing" : "legitimate"}`;
        badge.innerText = prediction;

        const sender = document.createElement("span");
        sender.innerText = `From: ${item.sender || "Unknown"}`;

        const confidence = document.createElement("div");
        confidence.className = "result-meta";
        confidence.innerText = `Confidence: ${confidencePct}`;

        const received = document.createElement("div");
        received.className = "result-meta";
        received.innerText = `Received: ${item.received_at || "-"}`;

        meta.appendChild(badge);
        meta.appendChild(sender);

        card.appendChild(subject);
        card.appendChild(meta);
        card.appendChild(confidence);
        card.appendChild(received);
        if (resultListEl) {
            resultListEl.appendChild(card);
        }
    });
}

async function checkLatestInboxEmails() {
    try {
        const baseUrl = await getBackendBaseUrl();
        setStatus("Syncing inbox...");
        clearResults();

        const syncResponse = await fetch(`${baseUrl}/api/sync-inbox?limit=5`, {
            method: "POST"
        });

        if (!syncResponse.ok) {
            throw new Error(`Sync failed (HTTP ${syncResponse.status})`);
        }

        const syncData = await syncResponse.json();
        if (syncData.error) {
            throw new Error(syncData.error);
        }

        setStatus("Loading latest scan results...");

        const activityResponse = await fetch(`${baseUrl}/api/activity?limit=50`);
        if (!activityResponse.ok) {
            throw new Error(`Activity load failed (HTTP ${activityResponse.status})`);
        }

        const activityData = await activityResponse.json();
        const items = Array.isArray(activityData.items) ? activityData.items : [];

        // Show the latest 5 scanned emails regardless of source.
        // Emails scanned via local server or /predict have null source_uid
        // and source_mailbox, so filtering by those fields hides everything.
        const latestFive = items.slice(0, 5);
        renderResults(latestFive);
    } catch (error) {
        setStatus(`Error connecting to backend: ${error.message}`);
    }
}
