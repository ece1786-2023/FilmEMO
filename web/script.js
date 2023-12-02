document.getElementById('submitBtn').addEventListener('click', fetchFilmData);

function fetchFilmData() {
    document.getElementById('filmData').innerHTML = '';
    var filmName = document.getElementById('filmName').value;
    var data = { film_name: filmName };

    // 显示并初始化进度条
    var progressBar = document.getElementById('progressBar');
    progressBar.style.display = 'block';
    progressBar.style.width = '0%';
    progressBar.innerText = '0%';

    // 模拟进度
    var progress = 0;
    var totalTime = 70000; // 总时间为70秒
    var intervalTime = 1000; // 每1秒更新一次进度
    var progressIncrement = 100 * (intervalTime / totalTime); // 每次增加的百分比

    var interval = setInterval(() => {
        progress += progressIncrement;
        if (progress > 100) progress = 100; // 确保不超过100%
        progressBar.style.width = progress + '%';
        progressBar.innerText = "Analyzing... " + Math.round(progress) + '%';

        if (progress >= 100) clearInterval(interval); // 到达100%时停止
    }, intervalTime);

    fetch('http://127.0.0.1:5000/filmEmo', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        // 立即设置进度条为100%
        clearInterval(interval);
        progressBar.style.width = '100%';
        progressBar.innerText = 'Analyzing Complete!';

        setTimeout(() => {
            progressBar.style.display = 'none';
        }, 3000); // 延迟隐藏进度条

        document.getElementById('filmData').innerHTML = '';
        displayFilmData(data);
    })
    .catch(error => {
        console.error('Error:', error);
        clearInterval(interval);
        progressBar.style.display = 'none';
    });
}


function displayFilmData(data) {
    var filmDataDiv = document.getElementById('filmData');
    var filmData = data.film_data;

    filmDataDiv.innerHTML = `
        <img src="${filmData.file_data}" alt="电影图片" style="max-width: 100%; height: auto; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <h2>${filmData.film_name}</h2>
        <div class="info-card">
            <div class="info-title">Director</div>
            <div class="info-content">${filmData.film_director}</div>
        </div>
        <div class="info-card">
            <div class="info-title">Film Type</div>
            <div class="info-content">${filmData.film_type}</div>
        </div>
        <div class="info-card">
            <div class="info-title">Film Length</div>
            <div class="info-content">${filmData.film_length}</div>
        </div>
        <div class="score-container">
            <div class="score-card">
                <h3>Audience Average Score</h3>
                <p>${data.audiences_average_score}/10</p>
            </div>
            <div class="score-card">
                <h3>Critics Average Score</h3>
                <p>${data.critics_average_score}/10</p>
            </div>
            <div class="score-card">
                <h3>Weighted Average Score</h3>
                <p>${data.average_score}/10</p>
            </div>
        </div>
    `;
}
