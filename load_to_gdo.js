#!/usr/bin/env node

const GDO_CONTEXT = 'testing';
const http = require('http');

http.request({
    host: 'dsigdo' + GDO_CONTEXT + '.doc.ic.ac.uk',
    path: '/api/GDO/ClearCave'
}, function(res){}).end();

console.log("Cleared cave.")

title_url = "https://gdo-cnn-visualization.herokuapp.com/title";
input_url = "https://gdo-cnn-visualization.herokuapp.com/input/";
structure_url = "https://gdo-cnn-visualization.herokuapp.com/network-structure";
layer_info_url = "https://gdo-cnn-visualization.herokuapp.com/info/";
predictions_url = "https://gdo-cnn-visualization.herokuapp.com/predictions/";
gradcam_url = "https://gdo-cnn-visualization.herokuapp.com/grad-cam/";
filter_url = "https://gdo-cnn-visualization.herokuapp.com/visualize/";


function post_title() {
    http.request({
        method: 'POST',
        headers: {'Cache-Control': 'no-cache',
                  'Content-Type': 'application/x-www-form-urlencoded'},
        host: 'dsigdo' + GDO_CONTEXT + '.doc.ic.ac.uk',
        path: '/api/Section/CreateAndDeploy?colStart=0&rowStart=0&width=5&height=1&appName=StaticHTML'},
        function(res){
            console.log('Status: ' + res.statusCode.toString().replace('200', 'OK'));
        }).end('=' + JSON.stringify({"url": title_url,
                                     "responsiveMode": true}));
}

function post_structure() {
    http.request({
        method: 'POST',
        headers: {'Cache-Control': 'no-cache',
                  'Content-Type': 'application/x-www-form-urlencoded'},
        host: 'dsigdo' + GDO_CONTEXT + '.doc.ic.ac.uk',
        path: '/api/Section/CreateAndDeploy?colStart=2&rowStart=1&width=3&height=3&appName=StaticHTML'},
        function(res){
            console.log('Status: ' + res.statusCode.toString().replace('200', 'OK'));
        }).end('=' + JSON.stringify({"url": structure_url,
                                     "responsiveMode": true}));
}

function post_input(label) {
    http.request({
        method: 'POST',
        headers: {'Cache-Control': 'no-cache',
                  'Content-Type': 'application/x-www-form-urlencoded'},
        host: 'dsigdo' + GDO_CONTEXT + '.doc.ic.ac.uk',
        path: '/api/Section/CreateAndDeploy?colStart=0&rowStart=1&width=2&height=3&appName=StaticHTML'},
        function(res){
            console.log('Status: ' + res.statusCode.toString().replace('200', 'OK'));
        }).end('=' + JSON.stringify({"url": input_url+label,
                                     "responsiveMode": true}));
}

function post_predictions(label) {
    http.request({
        method: 'POST',
        headers: {'Cache-Control': 'no-cache',
                  'Content-Type': 'application/x-www-form-urlencoded'},
        host: 'dsigdo' + GDO_CONTEXT + '.doc.ic.ac.uk',
        path: '/api/Section/CreateAndDeploy?colStart=5&rowStart=2&width=1&height=2&appName=StaticHTML'},
        function(res){
            console.log('Status: ' + res.statusCode.toString().replace('200', 'OK'));
        }).end('=' + JSON.stringify({"url": predictions_url+label,
                                     "responsiveMode": true}));
}

function post_layer_info(layer) {
    http.request({
        method: 'POST',
        headers: {'Cache-Control': 'no-cache',
                  'Content-Type': 'application/x-www-form-urlencoded'},
        host: 'dsigdo' + GDO_CONTEXT + '.doc.ic.ac.uk',
        path: '/api/Section/CreateAndDeploy?colStart=5&rowStart=0&width=1&height=2&appName=StaticHTML'},
        function(res){
            console.log('Status: ' + res.statusCode.toString().replace('200', 'OK'));
        }).end('=' + JSON.stringify({"url": layer_info_url+layer,
                                     "responsiveMode": true}));
}

function post_gradcam(label) {
    http.request({
        method: 'POST',
        headers: {'Cache-Control': 'no-cache',
                  'Content-Type': 'application/x-www-form-urlencoded'},
        host: 'dsigdo' + GDO_CONTEXT + '.doc.ic.ac.uk',
        path: '/api/Section/CreateAndDeploy?colStart=6&rowStart=0&width=2&height=4&appName=StaticHTML'},
        function(res){
            console.log('Status: ' + res.statusCode.toString().replace('200', 'OK'));
        }).end('=' + JSON.stringify({"url": gradcam_url+label,
                                     "responsiveMode": true}));
}

function post_filters(label, layer) {
    col = 8;
    row = 0;
    for (var i = 0; i < 32; i++) {
        http.request({
            method: 'POST',
            headers: {'Cache-Control': 'no-cache',
                      'Content-Type': 'application/x-www-form-urlencoded'},
            host: 'dsigdo' + GDO_CONTEXT + '.doc.ic.ac.uk',
            path: '/api/Section/CreateAndDeploy?colStart=' + col +
                  '&rowStart=' + row + '&width=1&height=1&appName=StaticHTML'
        },
        function(res){
            console.log('Status: ' + res.statusCode.toString().replace('200', 'OK'));
        }).end('=' + JSON.stringify({"url": filter_url+label+'/'+layer+'/'+i,
                                     "responsiveMode": true}));

        row++;
        if (row == 4) {
            row = 0;
            col++;
        }
    }
}


// var label = process.argv[2];
// var layer = process.argv[3];

var label = "wine";
var layer = "block4_conv1"

console.log(label);
console.log(layer);

post_title()
post_input(label)
post_structure()
post_layer_info(layer)
post_predictions(label)
post_gradcam(label)
post_filters(label, layer)



