//
//  main.swift
//  Neural Network 2
//
//  Created by Michael Bühlmann on 20.12.18.
//  Copyright © 2018 Michael Bühlmann. All rights reserved.
//

import Foundation
import Cocoa

postfix operator ^
infix operator  **

class Matrix: CustomStringConvertible {
    internal var data: [[Double]]
    
    var rows: Int
    var columns: Int
    
    init(_ data:[[Double]]) {
        self.data = data
        self.rows = data.count
        self.columns = data.first?.count ?? 0
    }
    
    init(_ data:[[Double]], rows:Int, columns:Int) {
        self.data = data
        self.rows = rows
        self.columns = columns
    }
    
    init(rows:Int, columns:Int) {
        self.data = [[Double]](repeating: [Double](repeating: 0, count: columns), count: rows)
        self.rows = rows
        self.columns = columns
    }
    
    /* init(_ list: [Double]) {
     self.data = [list]
     self.rows = list.count
     self.columns = list.count
     }*/
    
    subscript(row: Int, column: Int) -> Double {
        get {
            return data[row][column]
        }
        set {
            data[row][column] = newValue
        }
    }
    
    var dimensions: (rows: Int, columns: Int) {
        get {
            return (data.count, data.first?.count ?? 0)
        }
    }
    
    var rowCount: Int {
        get {
            return data.count
        }
    }
    
    var columnCount: Int {
        get {
            return data.first?.count ?? 0
        }
    }
    
    var count: Int {
        get {
            return rows * columns
        }
    }
    
    var description: String {
        var dsc = ""
        for row in 0..<rows {
            for col in 0..<columns {
                let d = data[row][col]
                dsc += String(d) + " "
            }
            dsc += "\n"
        }
        dsc += "rows: \(rows) columns: \(columns)\n"
        return dsc
    }
    
    func zeichne() {
        let context = NSGraphicsContext.current?.cgContext;
        //let shape = "square"
        context!.setLineWidth(1.0)
        
        for col in 0..<self.columns {
            for row in 0..<self.rows {
                let color = CGFloat(1.0/(7.0-self[row,col]));
                //print(color)
                context!.setStrokeColor(red: color, green: color, blue: color, alpha: 1)
                context!.setFillColor(red: color, green: color, blue: color, alpha: 1)
                
                let rectangle = CGRect(x: col*10, y: row*10, width: 10, height: 10)
                context!.addRect(rectangle)
                context!.drawPath(using: .fillStroke)
            }
        }
    }
    
    func maximum() -> Int {
        var data = 0.0
        var index = 0
        for row in 0..<self.rows {
            if self[row,0] > data {
                data = self[row,0]
                index = row
            }
        }
        return index
    }
    
    static func +(left: Matrix, right: Matrix) -> Matrix {
        assert(left.dimensions == right.dimensions, "Cannot add matrices of different dimensions")
        let m = Matrix(left.data, rows: left.rows, columns: left.columns)
        for row in 0..<left.rows {
            for col in 0..<left.columns {
                m[row,col] += right[row,col]
            }
        }
        return m
    }
    
    static func +=(left: Matrix, right: Matrix) {
        assert(left.dimensions == right.dimensions, "Cannot add matrices of different dimensions")
        for row in 0..<left.rows {
            for col in 0..<left.columns {
                left[row,col] += right[row,col]
            }
        }
    }
    static func -(left: Matrix, right: Matrix) -> Matrix {
        assert(left.dimensions == right.dimensions, "Cannot add matrices of different dimensions")
        let m = Matrix(left.data, rows: left.rows, columns: left.columns)
        for row in 0..<left.rows {
            for col in 0..<left.columns {
                m[row,col] -= right[row,col]
            }
        }
        return m
    }
    
    static func -(left: Double, right: Matrix) -> Matrix {
        let m = Matrix(rows: right.rows, columns: right.columns)
        for row in 0..<right.rows {
            for col in 0..<right.columns {
                m[row,col] = left - right[row,col]
            }
        }
        return m
    }
    
    static func *(left: Matrix, right: Matrix) -> Matrix {
        assert(left.dimensions == right.dimensions, "Cannot add matrices of different dimensions")
        let m = Matrix(left.data, rows: left.rows, columns: left.columns)
        for row in 0..<left.rows {
            for col in 0..<left.columns {
                m[row,col] *= right[row,col]
            }
        }
        return m
    }
    
    static func *(left: Double, right: Matrix) -> Matrix {
        let m = Matrix(right.data, rows: right.rows, columns: right.columns)
        for row in 0..<right.rows {
            for col in 0..<right.columns {
                m[row,col] *= left
            }
        }
        return m
    }
    
    static func ==(left: Matrix, right: Matrix) -> Bool {
        if left.rows != right.rows {
            return false
        }
        if left.columns != right.columns {
            return false
        }
        for i in 0..<left.rows {
            for j in 0..<left.columns {
                if left[i,j] != right[i,j] {
                    return false
                }
            }
        }
        return true
    }
    
    static postfix func ^(m: Matrix) -> Matrix {
        let t = Matrix(rows:m.columns, columns:m.rows)
        for row in 0..<m.rows {
            for col in 0..<m.columns {
                t[col,row] = m[row,col]
            }
        }
        return t
    }
    
    static func **(left: Matrix, right: Matrix) -> Matrix {
        assert(left.columns == right.rows, "Two matricies can only be matrix mulitiplied if one has dimensions mxn & the other has dimensions nxp where m, n, p are in R")
        let C = Matrix(rows: left.rows, columns: right.columns)
        for i in 0..<left.rows {
            for j in 0..<right.columns {
                for k in 0..<right.rows {
                    C[i, j] += left[i, k] * right[k, j]
                }
            }
        }
        return C
    }
    
}

class NeuralNetwork {
    internal var inodes: Int
    internal var hnodes: Int
    internal var onodes: Int
    internal var lr: Double
    
    internal var wih: Matrix
    internal var who: Matrix
    
    internal var r: UInt64 = 0
    
    init(inputnodes: Int, hidddennodes: Int, outputnodes: Int, learningrate: Double) {
        self.inodes = inputnodes
        self.hnodes = hidddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        
        self.wih = Matrix(rows: self.hnodes, columns: self.inodes)
        self.who = Matrix(rows: self.onodes, columns: self.hnodes)
        //self.wih = Matrix([[0.9, 0.3, 0.4], [0.2, 0.8, 0.2], [0.1, 0.5, 0.6]])
        //self.who = Matrix([[0.3, 0.7, 0.5], [0.6, 0.5, 0.2], [0.8, 0.1, 0.9]])
        //print(wih)
        //print(who)
        
        for row in 0..<wih.rows {
            for col in 0..<wih.columns {
                arc4random_buf(&self.r, 8)
                wih[row,col] = (Double(self.r) / Double(UInt64.max)) - 0.5
            }
        }
        for row in 0..<who.rows {
            for col in 0..<who.columns {
                arc4random_buf(&self.r, 8)
                who[row,col] = (Double(self.r) / Double(UInt64.max)) - 0.5
            }
        }
        
        //print(wih)
        //print(who)
        //print("Euler: \(M_E)")
        //print("Result: \(sigmoid(0.6))")
    }
    
    func sigmoid(_ x: Double) -> Double {
        return 1.0 / (1.0 + exp(-x))
    }
    
    func activation_function(_ list: Matrix) -> Matrix {
        let result = Matrix(rows: list.rows, columns: list.columns)
        for i in 0..<list.rows {
            result[i,0] = sigmoid(list[i,0])
        }
        return result
    }
    
    func query(_ input_list: Matrix) -> Matrix {
        let inputs = input_list^
        //print(inputs)
        let hidden_inputs = wih**inputs
        //print(hidden_inputs)
        let hidden_outputs = activation_function(hidden_inputs)
        //print(hidden_outputs)
        
        let final_inputs = who**hidden_outputs
        //print(final_inputs)
        let final_outputs = activation_function(final_inputs)
        //print(final_outputs)
        return final_outputs
    }
    
    func train(input_list: Matrix, target_list: Matrix) {
        let inputs = input_list^
        let targets = target_list^
        
        let hidden_inputs = wih**inputs
        let hidden_outputs = activation_function(hidden_inputs)
        
        let final_inputs = who**hidden_outputs
        let final_outputs = activation_function(final_inputs)
        
        let output_errors = targets - final_outputs
        //print(output_errors)
        let hidden_errors = (who^)**output_errors
        //print(hidden_errors)
        
        self.who += self.lr * ((output_errors * final_outputs * (1.0 - final_outputs)) ** (hidden_outputs^))
        self.wih += self.lr * ((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)) ** (inputs^))
        //print(who)
        //print(wih)
    }
}


let input_nodes = 784
let hidden_nodes = 100
let output_nodes = 10

let learning_rate = 0.2

let n = NeuralNetwork(inputnodes: input_nodes, hidddennodes: hidden_nodes, outputnodes: output_nodes, learningrate: learning_rate)
//Zeichne(filename: "mnist_train_100")

let training_data_file = "mnist_train"
let DocumentDirURL1 = try! FileManager.default.url(for: .downloadsDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
let fileURL1 = DocumentDirURL1.appendingPathComponent(training_data_file).appendingPathExtension("csv")
print("FilePath: \(fileURL1.path)")

var training_data_list = ""
do {
    training_data_list = try String(contentsOf: fileURL1)
} catch let error as NSError {
    print(error)
}
var lines = training_data_list.split() { $0 == "\n" }.map { $0 }

var inputs = Matrix(rows: 1, columns: input_nodes)
var targets = Matrix(rows: 1, columns: output_nodes)
let epochs = 5

for j in 0..<epochs {
    for line in 0..<lines.count {
        var all_values = lines[line].split() { $0 == "," }.map { Int($0)! }
        if (line % 100 == 0) {
            print(j, " - ", line)
        }
        for i in 0..<input_nodes {
            inputs[0,i] = (Double(all_values[i+1]) / 255.0 * 0.99) + 0.01
        }
        //print(inputs)
        for i in 0..<output_nodes {
            targets[0,i] = 0.01
        }
        targets[0,all_values[0]] = 0.99
        //print(targets)
        n.train(input_list: inputs, target_list: targets)
    }
}

let test_data_file = "mnist_test"
let DocumentDirURL2 = try! FileManager.default.url(for: .downloadsDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
let fileURL2 = DocumentDirURL2.appendingPathComponent(test_data_file).appendingPathExtension("csv")
print("FilePath: \(fileURL2.path)")

var test_data_list = ""
do {
    test_data_list = try String(contentsOf: fileURL2)
} catch let error as NSError {
    print(error)
}
lines = test_data_list.split() { $0 == "\n" }.map { $0 }
//print(lines)
var scorecard = [Int]()

for line in 0..<lines.count {
    if (line % 100 == 0) {
        print(line)
    }
    var all_values = lines[line].split() { $0 == "," }.map { Int($0)! }
    var correct_label = Int(all_values[0])
    //print("\(correct_label) correct_label")
    
    for i in 0..<input_nodes {
        inputs[0,i] = (Double(all_values[i+1]) / 255.0 * 0.99) + 0.01
    }
    var outputs = Matrix(rows: output_nodes, columns: 1)
    outputs = n.query(inputs)
    //print(outputs)
    let label = outputs.maximum()
    
    //print("\(label) network's answer")
    
    if (label == correct_label) {
        scorecard.append(1)
    } else {
        scorecard.append(0)
    }
}

let sum = scorecard.reduce(0, +)
let performance = Double(sum) / Double(scorecard.count)
print("performance = \(performance)" )
