const API_URL = "http://127.0.0.1:5000/predict";

const form = document.getElementById('jobForm');
const resultBox = document.getElementById('result');
const errorBox = document.getElementById('error');

form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const payload = {
        title: document.getElementById('title').value.trim(),
        company: document.getElementById('company').value.trim(),
        location: document.getElementById('location').value.trim(),
        salary: document.getElementById('salary').value.trim(),
        description: document.getElementById('description').value.trim(),
    };

    try {
        const resp = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await resp.json();

        // ⬅️ VERY IMPORTANT: save result to browser storage
        localStorage.setItem("analysis_result", JSON.stringify({
            ...payload,
            prediction: data.prediction,
            probability: data.probability,
            label: data.label
        }));

        // redirect to result page
        window.location.href = "result.html";

    } catch (err) {
        console.log("ERROR:", err);
        alert("Backend not responding");
    }
});
