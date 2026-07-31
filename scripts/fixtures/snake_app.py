from flask import Flask, render_template_string

app = Flask(__name__)

# Snake game served as a single page.
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Snake Game</title>
    <style>
        body { 
            display: flex; 
            flex-direction: column;
            justify-content: center; 
             align-items: center; 
             height: 100vh; 
             margin: 0; 
             background-color: #1a1a2e; 
             color: white; 
             font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
             overflow: hidden;
        }
        .score { 
              margin-bottom: 10px;
             font-size: 32px; 
             font-weight: bold;
             text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        canvas { 
              border: 5px solid #e94562; 
              box-shadow: 0 0 20px rgba(0,0,0,0.8); 
              background-color: #16213e; 
              border-radius: 5px;
        }
        #overlay {
            position: fixed;
            top: 50%;
            left: 50%;
             transform: translate(-50%, -50%);
             background: rgba(0, 0, 0, 0.9);
             padding: 30px;
             border-radius: 15px;
             text-align: center;
             border: 2px solid #e94562;
             z-index: 10;
        }
        button {
              padding: 10px 20px; 
             cursor: pointer;
             background-color: #e94562;
             color: white;
             border: none;
             border-radius: 5px;
             font-size: 16px;
             font-weight: bold;
             margin-top: 15px;
        }
        button:hover {
              background-color: #ff5e78;
        }
    </style>
</head>
<body>
    <div class="score"> Score: <span id="score">0</span></div>
    <div id="overlay" style="display:none;">
        <h1 id="msg">GAME OVER</h1>
        <button onclick="location.reload()"> Play Again</button>
    </div>
    <canvas id="gameCanvas" width="400" height="400"></canvas>

    <script>
        const canvas = document.getElementById('gameCanvas');
         const ctx = canvas.getContext('2d');
         const scoreElement = document.getElementById('score');
         const overlay = document.getElementById('overlay');
         const msgElement = document.getElementById('msg');

         const box = 20;
         let score = 0;
         let gameActive = true;
        
         let snake = [
            {x: 160, y: 160}, 
            {x: 140, y: 160}, 
            {x: 120, y: 160}]
        ;
        
         let food = {
            x: Math.floor(Math.random() * 19 + 1) * box,
            y: Math.floor(Math.random() * 19 + 1) * box
        };

        let direction = 'RIGHT'; 
        // Buffer to prevent 180-degree turns in a single frame
        let nextDirection = 'RIGHT';

        document.addEventListener('keydown', function(e) {
            const key = e.key;
            
            if(key === 'ArrowLeft' && direction !== 'RIGHT') nextDirection = 'LEFT';
            else if(key === 'ArrowUp' && direction !== 'DOWN') nextDirection = 'UP';
            else if(key === 'ArrowRight' && direction !== 'LEFT') nextDirection = 'RIGHT';
            else if(key === 'ArrowDown' && direction !== 'UP') nextDirection = 'DOWN';
        });

        function draw() {
            if(!gameActive) return;

            direction = nextDirection;

            // Clear Canvas
            ctx.fillStyle = '#16213e';
            
            // Draw Snake
            for(let i = 0; i < snake.length; i++) {
                ctx.fillStyle = (i === 0) ? '#4ecca3' : '#39cf9a';
                ctx.fillRect(snake[i].x, snake[i].y, box, box);
            }

            // Draw Food
            ctx.fillStyle = '#e94562';
            ctx.shadowBlur = 10;
            ctx.shadowColor = "#e94562";
            ctx.fillRect(food.x, food.y, box, box);
            ctx.shadowBlur = 0; // Reset shadow

            let headX = snake[0].x;
            let headY = snake[0].y;

            if(direction === 'LEFT') headX -= box;
            if(direction === 'UP') headY -= box;
            if(direction === 'RIGHT') headX += box;
            if(direction === 'DOWN') headY += box;

            // Collision Detection (Walls and Self)
            if(headX < 0 || headX >= canvas.width || headY < 0 || headY >= canvas.height || collision({x: headX, y: headY}, snake)) {
                gameOver();
                return;
            }

            // Movement Logic
            if(headX === food.x && headY === food.y) {
                score++;
                scoreElement.innerHTML = score;
                snake.unshift({x: headX, y: headY});
                spawnFood();
            } else {
                snake.unshift({x: headX, y: headY});
                snake.pop();
            }
        }

        function spawnFood() {
            let newFoodX, newFoodY;
            let isOverlapping = true;
            
            while (isOverlapping) {
                newFoodX = Math.floor(Math.random() * 19 + 1) * box;
                newFoodY = Math.floor(Math.random() * 19 + 1) * box;
                isOverlapping = collision({x: newFoodX, y: newFoodY}, snake);
            }
            
            food = { x: newFoodX, y: newFoodY };
        }

        function collision(head, array) {
            for(let i = 0; i < array.length; i++) {
                if(head.x === array[i].x && head.y === array[i].y) return true;
            }
            return false;
        }

        function gameOver() {
            gameActive = false;
            overlay.style.display = 'block';
            msgElement.innerText = 'GAME OVER! Score: ' + score;
        }

        setInterval(draw, 100); // Slightly faster speed for better playability
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

