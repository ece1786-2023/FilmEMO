document.getElementById('submitBtn').addEventListener('click', fetchFilmData);

function fetchFilmData() {
    var filmName = document.getElementById('filmName').value;
    var data = { film_name: filmName };

    fetch('http://127.0.0.1:5000/filmEmo', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        displayFilmData(data);
    })
    .catch(error => console.error('Error:', error));
}

function displayFilmData(data) {
    var filmDataDiv = document.getElementById('filmData');
    filmDataDiv.innerHTML = `
        <img src="${data.movie_pic_url}" alt="电影图片">
        <p>观众平均分数：${data.audiences_average_score}/10</p>
        <p>影评人平均分数：${data.critics_average_score}/10</p>
        <p>总平均分数：${data.average_score}/10</p>
    `;
}
