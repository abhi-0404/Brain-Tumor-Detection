document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const form = document.getElementById('upload-form');
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('image');
  const initialContent = document.getElementById('drop-zone-initial');
  const stagedContent = document.getElementById('drop-zone-staged');
  const previewImage = document.getElementById('preview-image');
  const previewName = document.getElementById('preview-name');
  const submitBtn = document.getElementById('submit-btn');
  const btnLabel = submitBtn.querySelector('.btn-label');
  const btnLoader = document.getElementById('btn-loader');

  // Constants
  const MAX_FILE_SIZE = 8 * 1024 * 1024; // 8MB
  const VALID_TYPES = ['image/jpeg', 'image/png', 'image/bmp'];

  // --- Core Functions ---

  /**
   * Handles the file selection, validates it, and updates the UI staging area.
   */
  const handleFile = (file) => {
    if (!file) return resetUI();

    // 1. Validation
    if (!VALID_TYPES.includes(file.type)) {
      alert('Unsupported file type. Please upload a JPG, PNG, or BMP.');
      return resetUI();
    }
    if (file.size > MAX_FILE_SIZE) {
      alert('File is too large. Maximum size is 8MB.');
      return resetUI();
    }

    // 2. Update UI (Staging Area)
    // Clean up previous object URL if it exists to prevent memory leaks
    if (previewImage.src) {
      URL.revokeObjectURL(previewImage.src);
    }
    
    previewImage.src = URL.createObjectURL(file);
    previewName.textContent = file.name;
    previewName.title = file.name; // Tooltip for long names

    // Transition States
    initialContent.hidden = true;
    stagedContent.hidden = false;
    submitBtn.disabled = false;
  };

  /**
   * Resets the upload zone to its initial state.
   */
  const resetUI = () => {
    fileInput.value = ''; // Clear input
    
    if (previewImage.src) {
      URL.revokeObjectURL(previewImage.src);
      previewImage.removeAttribute('src');
    }
    
    previewName.textContent = '';
    stagedContent.hidden = true;
    initialContent.hidden = false;
    submitBtn.disabled = true;
  };

  /**
   * Sets the UI into a loading state during form submission.
   */
  const setLoadingState = () => {
    submitBtn.disabled = true;
    submitBtn.classList.add('is-loading');
    btnLabel.textContent = 'Analyzing Scan...';
    btnLoader.hidden = false;
  };

  // --- Event Listeners ---

  // 1. File Input Change (Triggered via click or label association)
  fileInput.addEventListener('change', (e) => {
    handleFile(e.target.files[0]);
  });

  // 2. Drag and Drop Interaction
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
    document.body.addEventListener(eventName, preventDefaults, false);
  });

  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => {
      dropZone.classList.add('drag-active');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => {
      dropZone.classList.remove('drag-active');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      // Manually assign dropped file to the hidden input element
      const dataTransfer = new DataTransfer();
      dataTransfer.items.add(droppedFile);
      fileInput.files = dataTransfer.files;
      
      // Trigger handling
      handleFile(droppedFile);
    }
  });

  // 3. Form Submission
  form.addEventListener('submit', () => {
    // Only set loading state if a file exists
    if (fileInput.files.length > 0) {
      setLoadingState();
    }
  });
});