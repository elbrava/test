var today = new Date();


document.onload = function () {
    var time = document.getElementById("time")
    time.value = today.getHours() + ":" + today.getMinutes()
    var form1 = document.getElementById("form1")
    var form2 = document.getElementById("form2")
    var add = document.getElementById("add")
    var topic = document.getElementById("topic")
    var type = document.getElementById("type")
    var day = document.getElementById("day")


    add.onclick = function () {
        console.log("working");
        x = document.createElement("input");
        x.value = type.value + " " + topic.value + " " + day.value + " " + time.value
        form2.appendChild(x);
        type.value = "Lesson"
        topic.value = ""
        day.value = "Monday"
        time.value = ""
    

    }

}