document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const imagePreviewContainer = document.getElementById('image-preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-btn');
    const predictBtn = document.getElementById('predict-btn');
    const predictText = predictBtn.querySelector('.btn-text');
    const loader = predictBtn.querySelector('.loader');
    const resultsSection = document.getElementById('results-section');
    const predictedClass = document.getElementById('predicted-class');
    const predictedConfidence = document.getElementById('predicted-confidence');
    const barsContainer = document.getElementById('bars-container');
    const dummyWarning = document.getElementById('dummy-warning');
    const modelSelect = document.getElementById('model-select');

    let currentFile = null;

    // Trigger file input on click
    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation(); // Prevent triggering dropZone click
        fileInput.click();
    });

    dropZone.addEventListener('click', () => {
        if (!currentFile) fileInput.click();
    });

    // Handle Drag & Drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-active');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-active');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-active');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // Handle File Input Change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    // Remove Image
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        currentFile = null;
        fileInput.value = '';
        imagePreviewContainer.classList.add('hidden');
        imagePreview.src = '';
        predictBtn.disabled = true;
        resultsSection.classList.add('hidden');
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file.');
            return;
        }

        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreviewContainer.classList.remove('hidden');
            predictBtn.disabled = false;
            resultsSection.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }

    // Handle Prediction
    predictBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // UI State
        predictBtn.disabled = true;
        predictText.classList.add('hidden');
        loader.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        dummyWarning.classList.add('hidden');

        const formData = new FormData();
        formData.append('image', currentFile);
        formData.append('model', modelSelect.value);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }

            // Show Dummy Warning
            if (data.dummy) {
                dummyWarning.classList.remove('hidden');
                dummyWarning.querySelector('span').textContent = data.message;
            }

            // Populate Results
            predictedClass.textContent = data.prediction;
            predictedConfidence.textContent = (data.confidence * 100).toFixed(1) + '%';
            
            // Generate Bars
            barsContainer.innerHTML = '';
            
            // Sort predictions by confidence
            const sortedPreds = Object.entries(data.all_predictions)
                .sort((a, b) => b[1] - a[1]);

            // Top 5 predictions
            sortedPreds.slice(0, 5).forEach(([cls, conf]) => {
                const percentage = (conf * 100).toFixed(1);
                const isTop = cls === data.prediction;
                
                const barRow = document.createElement('div');
                barRow.className = 'bar-row';
                
                barRow.innerHTML = `
                    <div class="bar-label" style="color: ${isTop ? 'var(--text-main)' : 'var(--text-muted)'}">${cls}</div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width: 0%; ${isTop ? '' : 'background: var(--text-muted);'}"></div>
                    </div>
                    <div class="bar-value" style="color: ${isTop ? 'var(--accent-2)' : 'var(--text-muted)'}">${percentage}%</div>
                `;
                
                barsContainer.appendChild(barRow);
                
                // Animate the bar fill
                setTimeout(() => {
                    barRow.querySelector('.bar-fill').style.width = percentage + '%';
                }, 100);
            });

            resultsSection.classList.remove('hidden');

        } catch (error) {
            console.error(error);
            alert('An error occurred during prediction.');
        } finally {
            predictBtn.disabled = false;
            predictText.classList.remove('hidden');
            loader.classList.add('hidden');
        }
    });
});
