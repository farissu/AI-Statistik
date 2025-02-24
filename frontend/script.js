const API_BASE = "http://127.0.0.1:8000"; // Sesuaikan dengan backend Anda

// Kirim prompt ke API dan tampilkan kode Python yang dihasilkan
function sendPrompt() {
    let promptText = document.getElementById("promptInput").value;

    fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: promptText }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.script) {
            document.getElementById("outputCode").textContent = data.script;
        } else {
            alert("Gagal mendapatkan kode dari AI.");
        }
    })
    .catch(error => console.error("Error:", error));
}

// Copy kode ke clipboard
function copyCode() {
    let codeText = document.getElementById("outputCode").textContent;
    navigator.clipboard.writeText(codeText).then(() => {
        alert("Kode berhasil disalin!");
    }).catch(err => console.error("Gagal menyalin kode", err));
}

// Jalankan kode Python dengan API eksekusi
function runCode() {
    let codeText = document.getElementById("outputCode").textContent;

    fetch(`${API_BASE}/execute`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: codeText }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert("Error saat menjalankan kode: " + data.error);
        } else if (data.chart_data) {
            renderChart(data.chart_data);
        }
    })
    .catch(error => console.error("Error:", error));
}

// Render Chart.js dari data hasil eksekusi
function renderChart(chartData) {
    let ctx = document.getElementById("chartCanvas").getContext("2d");

    // Hapus grafik sebelumnya jika ada
    if (window.myChart) {
        window.myChart.destroy();
    }

    window.myChart = new Chart(ctx, {
        type: "bar",
        data: chartData,
        options: {
            responsive: true,
            scales: {
                y: { beginAtZero: true }
            }
        }
    });
}
