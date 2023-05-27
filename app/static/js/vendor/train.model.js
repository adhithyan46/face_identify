document.addEventListener('DOMContentLoaded', function() {
  const trainButton = document.getElementById('train-button');
  const messageContainer = document.getElementById('message-container');

  trainButton.addEventListener('click', function() {
    trainButton.disabled = true;
    messageContainer.textContent = 'Training in progress. Please wait...';

    fetch('/app/train_model/', { method: 'POST' })
      .then(response => response.json())
      .then(data => {
        messageContainer.textContent = 'Training completed!';
      })
      .catch(error => {
        messageContainer.textContent = 'Error occurred while training.';
      });
  });
});
