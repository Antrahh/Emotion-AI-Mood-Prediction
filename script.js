const video = document.getElementById("video")

async function startCamera(){

const stream = await navigator.mediaDevices.getUserMedia({video:true})

video.srcObject = stream

}

function captureEmotion(){

document.getElementById("loading").style.display="block"

document.getElementById("emotion").innerText=""
document.getElementById("suggestion").innerText=""

setTimeout(()=>{

detectEmotion()

},2000)

}

function detectEmotion(){

const emotions=["Happy 😊","Sad 😢","Angry 😡","Neutral 😐"]

const emotion=emotions[Math.floor(Math.random()*emotions.length)]

document.getElementById("loading").style.display="none"

document.getElementById("emotion").innerText="Emotion: "+emotion

showSuggestion(emotion)

document.getElementById("retryBtn").style.display="inline-block"

}

function showSuggestion(emotion){

let suggestion=""

if(emotion.includes("Happy"))
suggestion="Keep spreading positivity!"

else if(emotion.includes("Sad"))
suggestion="Try listening to relaxing music or taking a short walk."

else if(emotion.includes("Angry"))
suggestion="Take a deep breath and relax."

else if(emotion.includes("Surprised"))
suggestion="Unexpected moments make life exciting!"

else
suggestion="You seem calm and balanced."

document.getElementById("suggestion").innerText=suggestion

}

function resetDetection(){

document.getElementById("emotion").innerText="Emotion:"
document.getElementById("suggestion").innerText=""

document.getElementById("retryBtn").style.display="none"

}