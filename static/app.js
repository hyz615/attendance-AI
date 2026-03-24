/* Attendance AI — Frontend Application */

(function () {
    "use strict";

    // ── DOM refs ──
    const uploadArea = document.getElementById("uploadArea");
    const fileInput = document.getElementById("fileInput");
    const previewContainer = document.getElementById("previewContainer");
    const previewImage = document.getElementById("previewImage");
    const clearBtn = document.getElementById("clearBtn");
    const processBtn = document.getElementById("processBtn");
    const btnText = document.getElementById("btnText");
    const btnSpinner = document.getElementById("btnSpinner");

    const progressSection = document.getElementById("progressSection");
    const progressFill = document.getElementById("progressFill");
    const progressText = document.getElementById("progressText");

    const resultsSection = document.getElementById("resultsSection");
    const totalCount = document.getElementById("totalCount");
    const colCount = document.getElementById("colCount");
    const presentCount = document.getElementById("presentCount");
    const absentCount = document.getElementById("absentCount");
    const studentTableBody = document.getElementById("studentTableBody");

    const exportCsvBtn = document.getElementById("exportCsvBtn");
    const exportJsonBtn = document.getElementById("exportJsonBtn");
    const copyListBtn = document.getElementById("copyListBtn");
    const copyableList = document.getElementById("copyableList");
    const copyStatus = document.getElementById("copyStatus");

    const timingDetails = document.getElementById("timingDetails");
    const debugImages = document.getElementById("debugImages");

    const warningBanner = document.getElementById("warningBanner");
    const warningText = document.getElementById("warningText");
    const errorBanner = document.getElementById("errorBanner");
    const errorText = document.getElementById("errorText");

    let selectedFile = null;
    let lastResult = null;

    // ── Upload Handling ──

    uploadArea.addEventListener("click", () => fileInput.click());

    uploadArea.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadArea.classList.add("dragover");
    });

    uploadArea.addEventListener("dragleave", () => {
        uploadArea.classList.remove("dragover");
    });

    uploadArea.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadArea.classList.remove("dragover");
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFile(files[0]);
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
    });

    clearBtn.addEventListener("click", resetUpload);

    function handleFile(file) {
        if (!file.type.startsWith("image/")) {
            showError("Please select an image file.");
            return;
        }
        if (file.size > 20 * 1024 * 1024) {
            showError("File is too large (max 20 MB).");
            return;
        }

        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            uploadArea.style.display = "none";
            previewContainer.style.display = "block";
            processBtn.disabled = false;
            hideMessages();
        };
        reader.readAsDataURL(file);
    }

    function resetUpload() {
        selectedFile = null;
        fileInput.value = "";
        previewImage.src = "";
        uploadArea.style.display = "";
        previewContainer.style.display = "none";
        processBtn.disabled = true;
        resultsSection.style.display = "none";
        progressSection.style.display = "none";
        hideMessages();
    }

    // ── Process ──

    processBtn.addEventListener("click", processSheet);

    async function processSheet() {
        if (!selectedFile) return;

        setLoading(true);
        hideMessages();
        resultsSection.style.display = "none";
        showProgress(0, "Uploading image...");

        const formData = new FormData();
        formData.append("file", selectedFile);

        try {
            showProgress(20, "Sending to server...");

            const response = await fetch("/api/process", {
                method: "POST",
                body: formData,
            });

            showProgress(70, "Processing complete, loading results...");

            if (!response.ok) {
                const err = await response.json().catch(() => ({}));
                throw new Error(err.detail || `Server error (${response.status})`);
            }

            const data = await response.json();
            lastResult = data;

            showProgress(100, "Done!");
            setTimeout(() => {
                progressSection.style.display = "none";
                renderResults(data);
            }, 400);

        } catch (err) {
            showError(err.message);
            progressSection.style.display = "none";
        } finally {
            setLoading(false);
        }
    }

    // ── Render Results ──

    const studentTableHead = document.getElementById("studentTableHead");

    function updateSummaryCounts(grid) {
        let totalA = 0, totalP = 0;
        for (const row of grid) {
            for (const v of row) {
                if (v === "A") totalA++; else if (v === "P") totalP++;
            }
        }
        presentCount.textContent = totalP;
        absentCount.textContent = totalA;
    }

    function renderResults(data) {
        const totalCols = data.total_columns || 0;
        const grid = data.grid || [];
        const students = data.students || [];
        const info = data.student_info || [];

        // Compute full-grid stats
        let totalA = 0;
        let totalP = 0;
        for (const row of grid) {
            for (const v of row) {
                if (v === "A") totalA++; else if (v === "P") totalP++;
            }
        }
        totalCount.textContent = students.length;
        colCount.textContent = totalCols;
        presentCount.textContent = totalP;
        absentCount.textContent = totalA;

        // ─ Build table head ─
        studentTableHead.innerHTML = "";
        const headTr = document.createElement("tr");
        headTr.innerHTML = "<th>#</th><th>Surname</th><th>Given Name</th><th>Grade</th><th>OEN</th><th>DOB</th><th>Gender</th>";
        for (let c = 0; c < totalCols; c++) {
            const th = document.createElement("th");
            th.textContent = c + 1;
            headTr.appendChild(th);
        }
        studentTableHead.appendChild(headTr);

        // ─ Build table body ─
        studentTableBody.innerHTML = "";
        students.forEach((s, i) => {
            const tr = document.createElement("tr");
            const si = info[i] || {};

            // Parse name into surname + given
            const nameParts = (si.name || s.name || "").split(" ");
            const surname = nameParts[0] || "";
            const given = nameParts.slice(1).join(" ") || "";

            const tdNum = document.createElement("td");
            tdNum.textContent = i + 1;
            tr.appendChild(tdNum);

            const tdSur = document.createElement("td");
            tdSur.className = "cell-name";
            tdSur.textContent = surname;
            tr.appendChild(tdSur);

            const tdGiven = document.createElement("td");
            tdGiven.className = "cell-name";
            tdGiven.textContent = given;
            tr.appendChild(tdGiven);

            const tdGrade = document.createElement("td");
            tdGrade.textContent = si.grade || "";
            tr.appendChild(tdGrade);

            const tdOen = document.createElement("td");
            tdOen.textContent = si.oen || "";
            tr.appendChild(tdOen);

            const tdDob = document.createElement("td");
            tdDob.textContent = si.dob || "";
            tr.appendChild(tdDob);

            const tdGender = document.createElement("td");
            tdGender.textContent = si.gender || "";
            tr.appendChild(tdGender);

            // Each date column
            const rowGrid = grid[i] || [];
            for (let c = 0; c < totalCols; c++) {
                const td = document.createElement("td");
                td.className = "cell-status cell-editable";
                const val = rowGrid[c];
                if (val === "A") {
                    td.innerHTML = '<span class="mark-absent">A</span>';
                } else {
                    td.textContent = "";
                }
                // Click to toggle A/P
                td.dataset.row = i;
                td.dataset.col = c;
                td.addEventListener("click", function () {
                    const r = parseInt(this.dataset.row);
                    const cl = parseInt(this.dataset.col);
                    const curVal = grid[r][cl];
                    const newVal = curVal === "A" ? "P" : "A";
                    grid[r][cl] = newVal;
                    if (newVal === "A") {
                        this.innerHTML = '<span class="mark-absent">A</span>';
                    } else {
                        this.textContent = "";
                    }
                    updateSummaryCounts(grid);
                });
                tr.appendChild(td);
            }

            studentTableBody.appendChild(tr);
        });

        // Timing
        timingDetails.innerHTML = "";
        if (data.timing) {
            const stages = {
                preprocess: "Preprocessing",
                document_detection: "Document Detection",
                table_detection: "Table Detection",
                cell_extraction: "Cell Extraction",
                classification: "Classification",
                ocr_names: "OCR Names",
                aggregation: "Aggregation",
            };
            for (const [key, label] of Object.entries(stages)) {
                if (data.timing[key] != null) {
                    timingDetails.innerHTML += `
                        <div class="timing-item">
                            <span class="timing-label">${label}</span>
                            <span class="timing-value">${data.timing[key]}s</span>
                        </div>
                    `;
                }
            }
        }

        // Copyable student list — full grid
        let listText = "Attendance Grid (" + students.length + " students × " + totalCols + " columns)\n";
        listText += "─".repeat(60) + "\n";
        let hdr = "  #  Surname          Given Name      Gr  OEN        DOB        G";
        for (let c = 0; c < totalCols; c++) {
            hdr += (c + 1).toString().padStart(3);
        }
        listText += hdr + "\n";
        listText += "─".repeat(60) + "\n";
        students.forEach((s, i) => {
            const si = info[i] || {};
            const nameParts = (si.name || s.name || "").split(" ");
            const surname = (nameParts[0] || "").padEnd(16).substring(0, 16);
            const given = (nameParts.slice(1).join(" ") || "").padEnd(16).substring(0, 16);
            let line = (i + 1).toString().padStart(3) + "  " + surname + given;
            line += (si.grade || "").padEnd(4).substring(0, 4);
            line += (si.oen || "").padEnd(11).substring(0, 11);
            line += (si.dob || "").padEnd(11).substring(0, 11);
            line += (si.gender || "").padEnd(2).substring(0, 2);
            const rowGrid = grid[i] || [];
            for (let c = 0; c < totalCols; c++) {
                line += (rowGrid[c] || "-").padStart(3);
            }
            listText += line + "\n";
        });
        listText += "─".repeat(60) + "\n";
        listText += "Present: " + totalP + "  |  Absent: " + totalA + "\n";
        copyableList.value = listText;
        copyableList.rows = Math.min(Math.max(students.length + 6, 8), 45);

        // Debug images
        debugImages.innerHTML = "";
        (data.debug_images || []).forEach((url) => {
            const name = url.split("/").pop();
            const wrapper = document.createElement("div");
            wrapper.className = "debug-img-wrapper";
            wrapper.innerHTML = `
                <img src="${escapeHtml(url)}" alt="${escapeHtml(name)}" loading="lazy">
                <div class="caption">${escapeHtml(name)}</div>
            `;
            wrapper.addEventListener("click", () => openLightbox(url));
            debugImages.appendChild(wrapper);
        });

        if (data.warning) {
            showWarning(data.warning);
        }

        resultsSection.style.display = "";
    }

    // ── Exports ──

    exportCsvBtn.addEventListener("click", () => {
        if (!lastResult) return;
        const students = lastResult.students || [];
        const info = lastResult.student_info || [];
        const grid = lastResult.grid || [];
        const totalCols = lastResult.total_columns || 0;
        let csv = "Surname,Given Name,Grade,OEN,DOB,Gender";
        for (let c = 0; c < totalCols; c++) csv += "," + (c + 1);
        csv += "\n";
        students.forEach((s, i) => {
            const si = info[i] || {};
            const nameParts = (si.name || s.name || "").split(" ");
            const surname = nameParts[0] || "";
            const given = nameParts.slice(1).join(" ") || "";
            const esc = (v) => '"' + (v || "").replace(/"/g, '""') + '"';
            csv += esc(surname) + "," + esc(given) + "," + esc(si.grade) + "," + esc(si.oen) + "," + esc(si.dob) + "," + esc(si.gender);
            const row = grid[i] || [];
            for (let c = 0; c < totalCols; c++) csv += "," + (row[c] === "A" ? "A" : "");
            csv += "\n";
        });
        downloadText(csv, "attendance.csv", "text/csv");
    });

    exportJsonBtn.addEventListener("click", () => {
        if (!lastResult) return;
        const info = lastResult.student_info || [];
        const out = {
            students: (lastResult.students || []).map((s, i) => {
                const si = info[i] || {};
                return {
                    name: si.name || s.name,
                    grade: si.grade || "",
                    oen: si.oen || "",
                    dob: si.dob || "",
                    gender: si.gender || "",
                    attendance: (lastResult.grid || [])[i] || [],
                };
            }),
            total_columns: lastResult.total_columns || 0,
        };
        downloadText(JSON.stringify(out, null, 2), "attendance.json", "application/json");
    });

    copyListBtn.addEventListener("click", () => {
        if (!copyableList.value) return;
        copyableList.select();
        navigator.clipboard.writeText(copyableList.value).then(() => {
            copyStatus.textContent = "✓ Copied!";
            setTimeout(() => { copyStatus.textContent = ""; }, 2000);
        }).catch(() => {
            document.execCommand("copy");
            copyStatus.textContent = "✓ Copied!";
            setTimeout(() => { copyStatus.textContent = ""; }, 2000);
        });
    });

    function downloadText(text, filename, mime) {
        const blob = new Blob([text], { type: mime });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    // ── UI Helpers ──

    function setLoading(loading) {
        processBtn.disabled = loading;
        btnText.textContent = loading ? "Processing..." : "Process Sheet";
        btnSpinner.style.display = loading ? "" : "none";
    }

    function showProgress(pct, text) {
        progressSection.style.display = "";
        progressFill.style.width = pct + "%";
        progressText.textContent = text;
    }

    function showError(msg) {
        errorBanner.style.display = "";
        errorText.textContent = msg;
    }

    function showWarning(msg) {
        warningBanner.style.display = "";
        warningText.textContent = msg;
    }

    function hideMessages() {
        errorBanner.style.display = "none";
        warningBanner.style.display = "none";
    }

    function escapeHtml(str) {
        const div = document.createElement("div");
        div.textContent = str;
        return div.innerHTML;
    }

    // ── Lightbox ──

    function openLightbox(url) {
        const overlay = document.createElement("div");
        overlay.className = "lightbox";
        overlay.innerHTML = `<img src="${escapeHtml(url)}" alt="Debug visualization">`;
        overlay.addEventListener("click", () => overlay.remove());
        document.body.appendChild(overlay);
    }

})();
