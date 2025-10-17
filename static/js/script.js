// General utility functions for the health tracker

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Auto-fill today's datebahi or v acha se padhao
    //  in date fields
    const dateFields = document.querySelectorAll('input[type="date"]');
    dateFields.forEach(field => {
        if (!field.value) {
            field.valueAsDate = new Date();
        }
    });

    // Calculate BMI when height and weight are entered
    const heightInput = document.getElementById('height');
    const weightInput = document.getElementById('weight');
    const bmiInput = document.getElementById('bmi');

    if (heightInput && weightInput && bmiInput) {
        const calculateBMI = () => {
            const height = parseFloat(heightInput.value) / 100; // Convert cm to m
            const weight = parseFloat(weightInput.value);
            
            if (height > 0 && weight > 0) {
                const bmi = weight / (height * height);
                bmiInput.value = bmi.toFixed(1);
            }
        };

        heightInput.addEventListener('input', calculateBMI);
        weightInput.addEventListener('input', calculateBMI);
    }

    // Form validation enhancements
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(event) {
            const requiredFields = form.querySelectorAll('[required]');
            let isValid = true;

            requiredFields.forEach(field => {
                if (!field.value.trim()) {
                    isValid = false;
                    field.classList.add('is-invalid');
                } else {
                    field.classList.remove('is-invalid');
                }
            });

            if (!isValid) {
                event.preventDefault();
                event.stopPropagation();
                
                // Show alert for first invalid field
                const firstInvalid = form.querySelector('.is-invalid');
                if (firstInvalid) {
                    firstInvalid.focus();
                }
            }
        });
    });

    // Auto-dismiss alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 5000);
    });

    // Health tips display
    const healthTips = [
        "Remember to drink at least 8 glasses of water daily.",
        "Aim for 7-9 hours of quality sleep each night.",
        "Regular exercise can significantly improve your overall health.",
        "Monitor your vitals regularly to track your health progress.",
        "A balanced diet is key to maintaining good health.",
        "Don't forget to schedule regular health check-ups."
    ];

    // Display random health tip if tip container exists
    const tipContainer = document.getElementById('health-tip');
    if (tipContainer) {
        const randomTip = healthTips[Math.floor(Math.random() * healthTips.length)];
        tipContainer.textContent = randomTip;
    }
});

// Utility function to format dates
function formatDate(dateString) {
    const options = { year: 'numeric', month: 'short', day: 'numeric' };
    return new Date(dateString).toLocaleDateString(undefined, options);
}

// Function to export health data (placeholder for future implementation)
function exportHealthData() {
    alert('Export feature coming soon!');
}