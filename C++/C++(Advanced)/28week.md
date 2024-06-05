### 28주차: Web 개발

#### 강의 목표
- HTML, CSS, JavaScript 기초 이해 및 사용
- 프론트엔드 프레임워크 (React, Vue.js 등) 사용
- 백엔드 개발 (Node.js, Django, Flask 등) 이해 및 사용

#### 강의 내용

##### 1. HTML, CSS, JavaScript 기초
- **HTML 기초**

```html
<!DOCTYPE html>
<html>
<head>
    <title>My First HTML Page</title>
</head>
<body>
    <h1>Hello, World!</h1>
    <p>This is a paragraph.</p>
    <a href="https://www.example.com">Link to Example</a>
</body>
</html>
```

- **CSS 기초**

```css
body {
    font-family: Arial, sans-serif;
}

h1 {
    color: blue;
}

p {
    color: green;
}
```

- **JavaScript 기초**

```html
<!DOCTYPE html>
<html>
<head>
    <title>JavaScript Example</title>
</head>
<body>
    <h1 id="greeting">Hello, World!</h1>
    <button onclick="changeGreeting()">Change Greeting</button>

    <script>
        function changeGreeting() {
            document.getElementById("greeting").innerHTML = "Hello, JavaScript!";
        }
    </script>
</body>
</html>
```

##### 2. 프론트엔드 프레임워크
- **React 기본 사용법**

```jsx
// index.html
<!DOCTYPE html>
<html>
<head>
    <title>React Example</title>
</head>
<body>
    <div id="root"></div>
    <script src="https://unpkg.com/react/umd/react.development.js"></script>
    <script src="https://unpkg.com/react-dom/umd/react-dom.development.js"></script>
    <script src="https://unpkg.com/babel-standalone"></script>
    <script type="text/babel" src="app.js"></script>
</body>
</html>

// app.js
class App extends React.Component {
    render() {
        return (
            <div>
                <h1>Hello, React!</h1>
                <button onClick={() => alert('Button clicked!')}>Click Me</button>
            </div>
        );
    }
}

ReactDOM.render(<App />, document.getElementById('root'));
```

- **Vue.js 기본 사용법**

```html
<!DOCTYPE html>
<html>
<head>
    <title>Vue.js Example</title>
    <script src="https://cdn.jsdelivr.net/npm/vue@2"></script>
</head>
<body>
    <div id="app">
        <h1>{{ message }}</h1>
        <button @click="changeMessage">Change Message</button>
    </div>

    <script>
        new Vue({
            el: '#app',
            data: {
                message: 'Hello, Vue.js!'
            },
            methods: {
                changeMessage() {
                    this.message = 'Hello, Vue.js and JavaScript!';
                }
            }
        });
    </script>
</body>
</html>
```

##### 3. 백엔드 개발
- **Node.js 기본 사용법**

```javascript
// server.js
const http = require('http');

const hostname = '127.0.0.1';
const port = 3000;

const server = http.createServer((req, res) => {
    res.statusCode = 200;
    res.setHeader('Content-Type', 'text/plain');
    res.end('Hello, Node.js!\n');
});

server.listen(port, hostname, () => {
    console.log(`Server running at http://${hostname}:${port}/`);
});
```

- **Django 기본 사용법**

```bash
# 설치 및 프로젝트 생성
pip install django
django-admin startproject mysite

# views.py (mysite/mysite/views.py)
from django.http import HttpResponse

def index(request):
    return HttpResponse("Hello, Django!")

# urls.py (mysite/mysite/urls.py)
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index),
]

# 서버 실행
python manage.py runserver
```

- **Flask 기본 사용법**

```bash
# 설치
pip install flask

# app.py
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello, Flask!"

if __name__ == '__main__':
    app.run(debug=True)
```

#### 과제

1. **HTML, CSS, JavaScript를 사용하여 간단한 웹 페이지를 만드세요.**

```html
<!DOCTYPE html>
<html>
<head>
    <title>Simple Web Page</title>
    <style>
        body {
            font-family: Arial, sans-serif;
        }
        h1 {
            color: blue;
        }
        p {
            color: green;
        }
    </style>
</head>
<body>
    <h1 id="greeting">Hello, World!</h1>
    <p>This is a paragraph.</p>
    <button onclick="changeGreeting()">Change Greeting</button>

    <script>
        function changeGreeting() {
            document.getElementById("greeting").innerHTML = "Hello, JavaScript!";
        }
    </script>
</body>
</html>
```

2. **React를 사용하여 버튼 클릭 시 경고 메시지를 표시하는 웹 애플리케이션을 만드세요.**

```jsx
// index.html
<!DOCTYPE html>
<html>
<head>
    <title>React Example</title>
</head>
<body>
    <div id="root"></div>
    <script src="https://unpkg.com/react/umd/react.development.js"></script>
    <script src="https://unpkg.com/react-dom/umd/react-dom.development.js"></script>
    <script src="https://unpkg.com/babel-standalone"></script>
    <script type="text/babel" src="app.js"></script>
</body>
</html>

// app.js
class App extends React.Component {
    render() {
        return (
            <div>
                <h1>Hello, React!</h1>
                <button onClick={() => alert('Button clicked!')}>Click Me</button>
            </div>
        );
    }
}

ReactDOM.render(<App />, document.getElementById('root'));
```

3. **Node.js를 사용하여 "Hello, Node.js!" 메시지를 반환하는 서버를 만드세요.**

```javascript
// server.js
const http = require('http');

const hostname = '127.0.0.1';
const port = 3000;

const server = http.createServer((req, res) => {
    res.statusCode = 200;
    res.setHeader('Content-Type', 'text/plain');
    res.end('Hello, Node.js!\n');
});

server.listen(port, hostname, () => {
    console.log(`Server running at http://${hostname}:${port}/`);
});
```

4. **Flask를 사용하여 "Hello, Flask!" 메시지를 반환하는 웹 애플리케이션을 만드세요.**

```python
# app.py
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello, Flask!"

if __name__ == '__main__':
    app.run(debug=True)
```

#### 퀴즈

1. **HTML 태그 중 텍스트를 굵게 표시하는 태그는?**
   - A) `<i>`
   - B) `<b>`
   - C) `<u>`
   - D) `<em>`

2. **CSS에서 클래스 선택자를 사용하여 스타일을 적용하는 문법은?**
   - A) `#classname`
   - B) `.classname`
   - C) `classname`
   - D) `*classname`

3. **JavaScript에서 함수 선언을 위한 키워드는?**
   - A) `function`
   - B) `def`
   - C) `func`
   - D) `method`

4. **Node.js에서 서버를 생성할 때 사용하는 모듈은?**
   - A) `fs`
   - B) `http`
   - C) `os`
   - D) `path`

#### 퀴즈 해설

1. **HTML 태그 중 텍스트를 굵게 표시하는 태그는?**
   - **정답: B) `<b>`**
     - 해설: `<b>` 태그는 텍스트를 굵게 표시하는 데 사용됩니다.

2. **CSS에서 클래스 선택자를 사용하여 스타일을 적용하는 문법은?**
   - **정답: B) `.classname`**
     - 해설: CSS에서 클래스 선택자는 점(.)을 사용하여 스타일을 적용합니다.

3. **JavaScript에서 함수 선언을 위한 키워드는?**
   - **정답: A) `function`**
     - 해설: JavaScript에서 함수를 선언할 때는 `function` 키워드를 사용합니다.

4. **Node.js에서 서버를 생성할 때 사용하는 모듈은?**
   - **정답: B) `http`**
     - 해설: Node.js에서 서버를 생성할 때는 `http` 모듈을 사용합니다.

다음 주차 강의 내용을 요청하시면, 29주차 강의 내용, 과제 및 퀴즈를 상세히 제공해드리겠습니다.