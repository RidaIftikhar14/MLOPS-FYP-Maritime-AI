// Default map settings
var map = L.map('map',{ zoomControl: false }).setView([40.93174, -70.99243], 7);
var osm=L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
	attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
	subdomains: 'abcd',
	maxZoom: 20
});
osm.addTo(map);

new L.Control.Zoom({ position: 'topleft' }).addTo(map);
// ///////////////////////////////////////////////////////////////////
let starting_speed_val = 1;
let right_btn = document.getElementById("right_arrow");
let left_btn = document.getElementById("left_arrow");
let middle_val = document.getElementById("middle_text_text")
let speed = 1000;
let timespeed = 10;

right_btn.addEventListener('click', async () => {
  if(starting_speed_val < 32){
    starting_speed_val *= 2;
    middle_val.innerHTML = starting_speed_val.toString() + 'x';
    speed /= 2;
    timespeed /= 2;
    clearInterval(intervalId1);
    clearInterval(intervalId2);
    clearInterval(intervalId3);
    clearInterval(intervalId4);
    intervalId1 = setInterval(Get_Vessel_Data, speed);
    intervalId2 = setInterval(Vessel_Display,speed);
    intervalId3 = setInterval(Check_Anomaly, speed);
    intervalId4 = setInterval(DisplayTime, timespeed);
  }
});
// /////////////////////////////////////////////////////////////////////

left_btn.addEventListener('click', async () => {
  if(starting_speed_val > 1){
    starting_speed_val /= 2;
    middle_val.innerHTML = starting_speed_val.toString() + 'x';
    speed *= 2;
    timespeed*= 2;
    clearInterval(intervalId1);
    clearInterval(intervalId2);
    clearInterval(intervalId3);
    clearInterval(intervalId4);
    intervalId1 = setInterval(Get_Vessel_Data, speed);
    intervalId2 = setInterval(Vessel_Display,speed);
    intervalId3 = setInterval(Check_Anomaly, speed);
    intervalId4 = setInterval(DisplayTime, timespeed);
  }
});


let static_information;
let ports_information;
const anomaly_count = [];

// Define array of options for predictive search
var options = [];

console.log('Fetching Static Data')
fetch('Static_Data')
    .then(response => response.json())
        .then(data => {
          static_information = data;
          console.log('Static Data Fetched');
          console.log('Fetching Port information')
          // console.log(data);
          for(var i = 0; i < Object.keys(data.MMSI).length ; i++)
          {
            options.push(data.MMSI[i].toString());
          }
          for (let i = 0; i < Object.keys(static_information.MMSI).length; i++) {
            let extra = []
            extra.push(parseInt(Object.values(static_information.MMSI)[i]));
            extra.push(0);
            anomaly_count.push(extra);
          }
          Ports_information();
        })
        .catch(error => console.error('Error fetching static data:', error));

function Ports_information()
{
  fetch('Port_info')
  .then(response => response.json())
      .then(data => {
        ports_information = data;
        console.log('Port information Fetched');
        for(var i = 0 ; ports_information.Country[i] != undefined ; i++)
        {
          if(ports_information.Country[i] != undefined && ports_information.Port[i] != undefined && ports_information.Longitude[i] != undefined && ports_information.Latitude[i] != undefined)
          {
            var port_marker_options = {
              Country: ports_information.Country[i],
              Port: ports_information.Port[i],
              Longitude: ports_information.Longitude[i],
              Latitude: ports_information.Latitude[i],
              clickable: true,
              draggable: false,
              icon: L.icon({ iconUrl: 'static\\images\\port_small_big.ico' ,iconSize: [25, 25]})
            }
            var port_marker = L.marker([ports_information.Latitude[i],ports_information.Longitude[i]],port_marker_options).addTo(map).bindPopup(ports_information.Port[i]);
          }
        }
         // console.log('Starting Sim')
      })
      .catch(error => console.error('Error fetching ports information:', error));
}


let buffer = [];
async function Get_Vessel_Data(){
  fetch('Current_Time_Information')
    .then(response => response.json())
    .then(data => {
      // console.log(data)
      buffer.push(data);
  })
  .catch(error => {
    // Handle error
    console.log(error);
  });
}
let intervalId1 = setInterval(Get_Vessel_Data, speed);


let markers = []
let mmsi_present = []


function toRadians(degrees) {
  return degrees * Math.PI / 180;
}

function toDegrees(radians) {
  return radians * 180 / Math.PI;
}

async function Vessel_Display()
{
  if(buffer.length > 0)
  {
    First_index = buffer.shift();
    for(var i = 0; i < Object.keys(First_index.MMSI).length ; i++)
    {
      var MMSI = First_index.MMSI[i];

      if (!mmsi_present.includes(MMSI))
      {
        // console.log(First_index)
        var lat = First_index.LAT[i];
        var lon = First_index.LON[i];
        var vesselName = First_index.VesselName[i];
        var cog = First_index.COG[i];
        var length = First_index.Length[i];
        var width = First_index.Width[i];
        var draft = First_index.Draft[i];
        var vesselType = First_index.VesselType[i];
        // Define marker icon properties
        var temp_icon_str = 'static\\images\\';
        if (vesselType === 70)
        {
          temp_icon_str = temp_icon_str + 'vessel_70_big.ico';
        }
        else if (vesselType === 80)
        {
          temp_icon_str = temp_icon_str + 'vessel_80_big.ico';
        }
        else if (vesselType === 60)
        {
          temp_icon_str = temp_icon_str + 'vessel_60_big.ico';
        }
        else if (vesselType === 30)
        {
          temp_icon_str = temp_icon_str + 'vessel_30_big.ico';
        }
        else if (vesselType === 31)
        {
          temp_icon_str = temp_icon_str + 'vessel_31_big.ico';
        }
        else if (vesselType === 37)
        {
          temp_icon_str = temp_icon_str + 'vessel_37_big.ico';
        }
        else if (vesselType === 69)
        {
          temp_icon_str = temp_icon_str + 'vessel_69_big.ico';
        }
        else if (vesselType === 52)
        {
          temp_icon_str = temp_icon_str + 'vessel_52_big.ico';
        }
        else if (vesselType === 57)
        {
          temp_icon_str = temp_icon_str + 'vessel_57_big.ico';
        }
        else
        {
          temp_icon_str = temp_icon_str + 'vessel_default_big.ico';
        }

        let iconProps = {
          iconUrl: temp_icon_str,
          iconSize: [25, 25],
        };
        // console.log('New: '+MMSI);
        var temp_marker = L.marker([lat, lon], { icon: L.icon(iconProps), rotationAngle: 0}).addTo(map).bindPopup(
          `<b>${vesselName}</b><br>COG: ${cog}<br>Length: ${length}<br>Width: ${width}<br>Draft: ${draft}<br>lat: ${lat}<br>lon: ${lon}`
        );
        mmsi_present.push(MMSI);
        markers.push(temp_marker);
      }
      else
      {
        var lat = First_index.LAT[i];
        var lon = First_index.LON[i];
        var vesselName = First_index.VesselName[i];
        var cog = First_index.COG[i];
        var length = First_index.Length[i];
        var width = First_index.Width[i];
        var draft = First_index.Draft[i];
        
        // console.log('Old: '+MMSI);
        var temp_marker = markers[mmsi_present.indexOf(MMSI)];
        // Convert coordinates to radians
        const radLat1 = toRadians(temp_marker.getLatLng().lat);
        const radLon1 = toRadians(temp_marker.getLatLng().lng);
        const radLat2 = toRadians(lat);
        const radLon2 = toRadians(lon);
        const earthRadius = 6371;
        
        // Calculate great-circle distance using Haversine formula
        const dLon = radLon2 - radLon1;
        const dLat = radLat2 - radLat1;
        const a = Math.pow(Math.sin(dLat / 2), 2) +
                  Math.cos(radLat1) * Math.cos(radLat2) *
                  Math.pow(Math.sin(dLon / 2), 2);
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        const distance = earthRadius * c;

        // Calculate bearing (angle) between the two points
        const y = Math.sin(dLon) * Math.cos(radLat2);
        const x = Math.cos(radLat1) * Math.sin(radLat2) -
                  Math.sin(radLat1) * Math.cos(radLat2) * Math.cos(dLon);
        const bearing = toDegrees(Math.atan2(y, x));
        temp_marker.setRotationAngle(bearing);
        temp_marker.setLatLng(L.latLng(lat, lon));
      }
    }
  }
}

let intervalId2 = setInterval(Vessel_Display,speed);

// Anomally Check
// let Anomally_log = []


var big_alarm = 0;
var small_alarm = 0;


async function Check_Anomaly()
{
  for (let i = 0; i < anomaly_count.length; i++) {
    if(anomaly_count[i][1] >10)
    {
      anomaly_count[i][1] = 0;
      document.getElementsByClassName("anomaly_detection_issue_big_value")[0].innerHTML = big_alarm;
      big_alarm+=1;
    }
  }
  fetch('ai_models')
  .then(response => response.json())
  .then(data => {
    console.log(data);
    // console.log(Object.keys(data[0]).length)
    // Anomaly_log_buffer.push(data);
    // console.log(anomaly_count.length);
    for (let temp_var=0 ; temp_var < Object.keys(data[0]).length ; temp_var++)
    {
      console.log
        for(let j = 0 ; j < Object.keys(data[2][temp_var]).length ; j++)
        {
          var temp_buffer = []
          var temp_data1 = data[0][temp_var];
          var temp_data2 = data[1][temp_var];

          if (j==0)
          { 
            if(data[2][temp_var][0] == -1)
            {
              // if(mmsi_present.includes(temp_data1))
              // {
              //   mmsi_present.indexOf(temp_data1)
              // }
              for (let i = 0; i < anomaly_count.length; i++) {
                if(anomaly_count[i][0] === temp_data1)
                {
                  anomaly_count[i][1] += 1;
                  document.getElementsByClassName("anomaly_detection_issue_small_value")[0].innerHTML = small_alarm;
                  small_alarm += 1;
                }
              }
              // console.log(anomaly_count);
              // console.log("loop done");
              temp_buffer.push(temp_data1);
              temp_buffer.push(temp_data2);
              temp_buffer.push("Route");
              // Anomally_log.push(temp_buffer);
              let newRow = document.createElement('tr');
              newRow.innerHTML = `
              <td>${temp_data1}</td>
              <td>${temp_data2}</td>
              <td>${"Route"}</td>
              `;
              // Add the new row to the table body
              let tableBody = document.querySelector('#myTable tbody');
              tableBody.appendChild(newRow);
            }
          }
          else if (j == 1)
          {
            if(data[2][temp_var][1] == -1)
            {
              for (let i = 0; i < anomaly_count.length; i++) {
                if(anomaly_count[i][0] === temp_data1)
                {
                  anomaly_count[i][1] += 1;
                  document.getElementsByClassName("anomaly_detection_issue_small_value")[0].innerHTML = small_alarm;
                  small_alarm += 1;
                }
              }
              temp_buffer.push(temp_data1);
              temp_buffer.push(temp_data2);
              temp_buffer.push("Speed");
              // Anomally_log.push(temp_buffer);
              let newRow = document.createElement('tr');
              newRow.innerHTML = `
              <td>${temp_data1}</td>
              <td>${temp_data2}</td>
              <td>${"Speed"}</td>
              `;
              // Add the new row to the table body
              let tableBody = document.querySelector('#myTable tbody');
              tableBody.appendChild(newRow);
            }
          }
          else
          {
            if(data[2][temp_var][2] == -1)
            {
              for (let i = 0; i < anomaly_count.length; i++) {
                if(anomaly_count[i][0] === temp_data1)
                {
                  anomaly_count[i][1] += 1;
                  document.getElementsByClassName("anomaly_detection_issue_small_value")[0].innerHTML = small_alarm;
                  small_alarm += 1;
                }
              }
              temp_buffer.push(temp_data1);
              temp_buffer.push(temp_data2);
              temp_buffer.push("Cargo");
              // Anomally_log.push(temp_buffer);
              let newRow = document.createElement('tr');
              newRow.innerHTML = `
              <td>${temp_data1}</td>
              <td>${temp_data2}</td>
              <td>${"Cargo"}</td>
              `;
              // Add the new row to the table body
              let tableBody = document.querySelector('#myTable tbody');
              tableBody.appendChild(newRow);
            }
          }

        // }

      }
    }

    // console.log(Anomally_log);
  })
  .catch(error => {
  // Handle error
  console.log(error);
  });
}

// When the user clicks on <div>, open the popup
let popup_val = 0;
function myFunction() {
  var popup = document.getElementById("myPopup");
  popup.classList.toggle("show");
  if(popup_val == 0)
  {
    document.getElementsByClassName("anomaly_detection_issue_big_value")[0].innerHTML = 0;
    document.getElementsByClassName("anomaly_detection_issue_small_value")[0].innerHTML = 0;
    popup_val = 1;
  }
  else
  {
    popup_val = 0;
  }
}

let intervalId3 = setInterval(Check_Anomaly, speed);


// Show predictive search options based on user input
function showPredictions(event) {
  var input = event.target.value;
  var predictionsDiv = document.getElementById("predictions");
  predictionsDiv.innerHTML = "";
  if (input.length > 0) {
    options.forEach(function(option) {
      if (option.toLowerCase().startsWith(input.toLowerCase())) {
        var prediction = document.createElement("div");
        prediction.className = "prediction";
        prediction.textContent = option;
        prediction.onclick = function() {
          document.getElementById("search-box").value = option;
          predictionsDiv.innerHTML = "";
        };
        predictionsDiv.appendChild(prediction);
      }
    });
  }
}

// Send search query to Flask server
async function sendSearchQuery() {
  var searchBox = document.getElementById("search-box");
  var query = searchBox.value;
  if (query.length > 0) {
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/search", true);
    xhr.setRequestHeader('Content-Type', 'text/plain');
    xhr.onreadystatechange = function() {
      if (xhr.readyState === 4 && xhr.status === 200) {
        showPopup(xhr.responseText);
      }
    };
    xhr.send(query);
  }
}

// Show popup with server response
async function showPopup(content) 
{
  var popup = document.getElementById("popup");
  var popupContent = document.getElementById("popup-content");
  var popup_before_content = document.getElementById("before_popup_content");
  var popup_close = document.getElementById("crossmark");
  // console.log(content);
  var obj = JSON.parse(content);
  var contentMMSI = `MMSI:   ${obj.MMSI}`;
  var contentHeading = `Heading:   ${obj.Heading}`;
  var contentName = `Vessel Name:   ${obj.VesselName}`;
  var contentIMO = `IMO:   ${obj.IMO}`;
  var contentCallsign =  `CallSign:   ${obj.CallSign}`;
  var contentType =  `VesselType:   ${obj.VesselType}`;
  var contentStatus =  `Status:   ${obj.Status}`;
  var contentLength =  `Length:   ${obj.Length}`;
  var contentWidth =  `Width:   ${obj.Width}`;
  var contentDraft =  `Draft:   ${obj.Draft}`;
  // var contentPosition =  `Position:   ${obj.Position}`;
  var img_address1 = `${obj.address_one}` + '.jpg';
  var img_address2 = `${obj.address_two}` + '.jpg';


  // console.log(mmsi_present);
  // console.log(parseInt(obj.MMSI));
  let check = mmsi_present.indexOf(parseInt(obj.MMSI));
  // console.log(check);
  if (check != -1)
  {
    markers[mmsi_present.indexOf(parseInt(obj.MMSI))].openPopup();
  }
  

  popup.style.display = "block";
  var output = '<div style="text-align:center;"><img id="popup-img" src="' + img_address2 + '" style="max-width:100%; margin-top:-10px; border-radius: 10px; max-height:100%;"><a id="popup-img-prev" href="#" style="position:absolute; left:20px; top:25%; transform:translateY(-50%); font-size:20px; color:#000000;">&#10094;</a><a id="popup-img-next" href="#" style="position:absolute; right:20px; top:25%; transform:translateY(-50%); font-size:20px; color:#000000;;">&#10095;</a></div>' + contentName + '<br>' + contentIMO + '<br>' + contentCallsign + '<br>' + contentType + '<br>' + contentStatus + '<br>' + contentLength + '<br>' + contentWidth + '<br>' + contentHeading + '<br>' + contentDraft + '<br>';

  popupContent.innerHTML = output;
  popup_before_content.style.display = "none";
  popup_close.style.display = "block";
  popupContent.style.fontSize = "17px";
  popupContent.style.fontFamily = 'LEMON MILK LIGHT';
  if (theme_id==1)
  {
    popupContent.style.color = "#e1e1e1";
  } 
  else
  {
    popupContent.style.color = "#262626";
  }

  fetch(img_address2, { method: 'HEAD' })
  .then(response => {
    if (response.ok) {
      // console.log('Image exists');
      popup.style.top = "385px";
    } else {
      // console.log('Image does not exist');
      popup.style.top = "300px";
    }
  })
  // popup.style.top = "385px";
  popup.style.borderRadius = "20px";

  var popupImg = document.getElementById("popup-img");
  var popupImgPrev = document.getElementById("popup-img-prev");
  var popupImgNext = document.getElementById("popup-img-next");

  // var imgArray = [img_address2, img_address1];
  var imgArray = [img_address2];
  var currentImgIndex = 0;

  popupImgPrev.onclick = function() {
    currentImgIndex--;
    if (currentImgIndex < 0) {
      currentImgIndex = imgArray.length - 1;
    }
    popupImg.src = imgArray[currentImgIndex];
  };

  popupImgNext.onclick = function() {
    currentImgIndex++;
    if (currentImgIndex >= imgArray.length) {
      currentImgIndex = 0;
    }
    popupImg.src = imgArray[currentImgIndex];
  };
}
  // Hide popup
function hidePopup() {
  var popup = document.getElementById("popup");
  var popup_before_content = document.getElementById("before_popup_content");
  var popup_close = document.getElementById("crossmark");
  var popupContent = document.getElementById("popup-content");
  popupContent.textContent = "";
  popup_before_content.style.display = "block";
  popup_close.style.display = "none";
  popup.style.top = "160px";
  
  var searchBox = document.getElementById("search-box");
  searchBox.value = "";

  popup.style.display = "none";
}


var theme_id=1;
let [milliseconds,seconds,minutes,hours] = [0,0,0,0];
let timerRef = document.getElementById('timer')



async function change_theme()
{
  let navbar = document.getElementsByClassName("navbar")[0];
  let themelabel = document.getElementById("theme_label");
  let searchboxtext = document.getElementById("searchboxtext");
  let searchbox = document.getElementById("search-box");
  let predictions = document.getElementById("predictions");
  let searchbtn = document.getElementById("searchbtn");
  let popup = document.getElementById("popup");
  var popupContent = document.getElementById("popup-content");
  let timer_bg = document.getElementById("timer_bg");
  let timer = document.getElementById("timer");
  let left_arrow = document.getElementById("left_arrow");
  let middle_text = document.getElementById("middle_text");
  let middle_text_text = document.getElementById("middle_text_text");
  let right_arrow = document.getElementById("right_arrow");
  let triangleR = document.getElementsByClassName("triangle-right")[0];
  let triangleL = document.getElementsByClassName("triangle-left")[0];
  let information_card = document.getElementById("information_card");
  let gallerytext0 = document.getElementsByClassName("gallery-item-text")[0];
  let gallerytext1 = document.getElementsByClassName("gallery-item-text")[1];
  let gallerytext2 = document.getElementsByClassName("gallery-item-text")[2];
  let gallerytext3 = document.getElementsByClassName("gallery-item-text")[3];
  let gallerytext4 = document.getElementsByClassName("gallery-item-text")[4];
  let gallerytext5 = document.getElementsByClassName("gallery-item-text")[5];
  let gallerytext6 = document.getElementsByClassName("gallery-item-text")[6];
  let gallerytext7 = document.getElementsByClassName("gallery-item-text")[7];
  let gallerytext8 = document.getElementsByClassName("gallery-item-text")[8];
  let anomaly_detection_log = document.getElementById("anomaly_detection_log");
  let anomaly_detection_issue_big = document.getElementById("anomaly_detection_issue_big");
  let anomaly_detection_issue_small = document.getElementById("anomaly_detection_issue_small");
  let logtext = document.getElementById("LOG_text");
  let navlogo = document.getElementById("nav_logo");
  let navButton1 = document.getElementsByClassName("navbuttons")[0];
  let navButton2 = document.getElementsByClassName("navbuttons")[1];
  let anomaly_detection_popup = document.getElementById("myPopup");
  // JavaScript code
  let table = document.getElementById("myTable"); // Get the table element

  if(theme_id==0)
    {
        theme_id=1;
        osm=L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
            subdomains: 'abcd',
            maxZoom: 20
        });
        osm.addTo(map);

        navlogo.style.filter = "invert(0)";
        navbar.style.backgroundColor = "#262626";
        themelabel.style.color = "#262626";
        searchboxtext.style.color = "#262626";
        searchbox.style.color = "#262626";
        searchbox.style.border = "2px solid #262626"
        predictions.style.color = "#262626";
        predictions.style.backgroundColor = "#e1e1e1;"
        searchbtn.style.filter = "invert(0.8)";
        popup.style.backgroundColor = "#262626";
        popupContent.style.color = "#e1e1e1";
        timer_bg.style.background = "#262626";
        timer.style.color = "#e1e1e1";
        left_arrow.style.background = "#262626";
        middle_text.style.background = "#262626";
        middle_text_text.style.color = "#e1e1e1";
        right_arrow.style.background = "#262626";
        triangleR.style.borderLeft = "15px solid #e1e1e1";
        triangleL.style.borderRight = "15px solid #e1e1e1";
        information_card.style.backgroundColor = "#262626";
        gallerytext0.style.color = "#e1e1e1";
        gallerytext1.style.color = "#e1e1e1";
        gallerytext2.style.color = "#e1e1e1";
        gallerytext3.style.color = "#e1e1e1";
        gallerytext4.style.color = "#e1e1e1";
        gallerytext5.style.color = "#e1e1e1";
        gallerytext6.style.color = "#e1e1e1";
        gallerytext7.style.color = "#e1e1e1";
        gallerytext8.style.color = "#e1e1e1";
        anomaly_detection_log.style.backgroundColor = "#262626";
        anomaly_detection_issue_big.style.backgroundColor = "#262626";
        anomaly_detection_issue_small.style.backgroundColor = "#262626";
        logtext.style.color = "#e1e1e1";
        navButton1.style.color = "#e1e1e1";
        navButton2.style.color = "#e1e1e1";
        anomaly_detection_popup.style.backgroundColor = "#262626";
        anomaly_detection_popup.style.color = "#e1e1e1";

        // Loop through each row in the table
        for (let i = 0; i < table.rows.length; i++) 
        {
          // Loop through each cell in the row
          for (let j = 0; j < table.rows[i].cells.length; j++) {
            // Set the border color of the cell to red
            if (i!=0)
            {
              table.rows[i].cells[j].style.border = "1px solid whitesmoke";
            }
          }
        }
        
      }
    else
    {
        theme_id=0;
        osm=L.tileLayer('https://api.maptiler.com/maps/ch-swisstopo-lbm-dark/{z}/{x}/{y}.png?key=XQkD6E3c6Kx34EjiLcnX', {
            attribution: '<a href="https://www.maptiler.com/copyright/" target="_blank">&copy; MapTiler</a> <a href="https://www.openstreetmap.org/copyright" target="_blank">&copy; OpenStreetMap contributors</a> © swisstopo <a href="https://www.maptiler.com/copyright/" target="_blank">&copy; MapTiler</a> <a href="https://www.openstreetmap.org/copyright" target="_blank">&copy; OpenStreetMap contributors</a> <a href="https://www.swisstopo.admin.ch/en/home.html" target="_blank">&copy; swisstopo</a>'
        });
        osm.addTo(map);

        navlogo.style.filter = "invert(0.9)";
        navbar.style.backgroundColor = "#e1e1e1";
        themelabel.style.color = "#e1e1e1";
        searchboxtext.style.color = "#e1e1e1";
        searchbox.style.color = "#262626";
        searchbox.style.border = "2px solid #e1e1e1"
        predictions.style.color = "#262626";
        predictions.style.backgroundColor = "#e1e1e1";
        searchbtn.style.filter = "invert(0)";
        popup.style.backgroundColor = "#e1e1e1";
        popupContent.style.color = "#262626";
        timer_bg.style.background = "#e1e1e1";
        timer.style.color = "#262626";
        left_arrow.style.background = "#e1e1e1";
        middle_text.style.background = "#e1e1e1";
        middle_text_text.style.color = "#262626";
        right_arrow.style.background = "#e1e1e1";
        triangleR.style.borderLeft = "15px solid #262626";
        triangleL.style.borderRight = "15px solid #262626";
        information_card.style.backgroundColor = "#e1e1e1";
        gallerytext0.style.color = "#262626";
        gallerytext1.style.color = "#262626";
        gallerytext2.style.color = "#262626";
        gallerytext3.style.color = "#262626";
        gallerytext4.style.color = "#262626";
        gallerytext5.style.color = "#262626";
        gallerytext6.style.color = "#262626";
        gallerytext7.style.color = "#262626";
        gallerytext8.style.color = "#262626";
        anomaly_detection_log.style.backgroundColor = "#e1e1e1";
        anomaly_detection_issue_big.style.backgroundColor = "#e1e1e1";
        anomaly_detection_issue_small.style.backgroundColor = "#e1e1e1";
        logtext.style.color = "#262626";
        navButton1.style.color = "#262626";
        navButton2.style.color = "#262626";
        anomaly_detection_popup.style.backgroundColor = "#e1e1e1";
        anomaly_detection_popup.style.color = "#262626";

        // Loop through each row in the table
        for (let i = 0; i < table.rows.length; i++) 
        {
          // Loop through each cell in the row
          for (let j = 0; j < table.rows[i].cells.length; j++) {
            if (i!=0)
            {
              table.rows[i].cells[j].style.border = "1px solid black";
            }
          }
        }
      }
}

async function DisplayTime()
{
    milliseconds+=10;
    if(milliseconds == 1000)
    {
        milliseconds = 0;
        seconds++;
        if(seconds == 60)
        {
            seconds = 0;
            minutes++;
            if(minutes == 60)
            {
                minutes = 0;
                hours++;
            }
        }
    }

    let h = hours < 10 ? "0" + hours : hours;
    let m = minutes < 10 ? "0" + minutes : minutes;
    let s = seconds < 10 ? "0" + seconds : seconds;
    let ms = milliseconds < 10 ? "00" + milliseconds : milliseconds < 100 ? "0" + milliseconds : milliseconds;
    timerRef.innerHTML = ` ${h} : ${m} : ${s} : ${ms}`;
}

let intervalId4 = setInterval(DisplayTime, timespeed);


