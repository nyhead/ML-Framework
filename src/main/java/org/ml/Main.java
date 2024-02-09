package org.ml;

import mikera.matrixx.Matrix;
import mikera.vectorz.Vector;

import java.io.*;
import java.util.*;

public class Main {
    public static void main(String[] args) throws Exception {
        String csvFile = "D:/winter-2023/java_related/ML-Framework/heart.csv";
        BufferedReader br = new BufferedReader(new FileReader(csvFile));
        String line;
        ArrayList<Double> targets = new ArrayList<>();
        ArrayList<ArrayList<Double>> train = new ArrayList<>();
        List<String> data = new ArrayList<>();

        while ((line = br.readLine()) != null) {
            String[] values = line.split(",");
            ArrayList<Double> trainRow = new ArrayList<>();
            // Assuming the last column is the target
            for (int i = 0; i < values.length - 1; i++) {
                trainRow.add(Double.valueOf(values[i]));
            }
            targets.add(Double.parseDouble(values[values.length -  1]));

            // The rest of the columns are the data
            StringJoiner joiner = new StringJoiner(" ", "", "\n");
            for (int i = 0; i < values.length - 1; i++) {
                joiner.add(values[i]);
            }
            data.add(joiner.toString());

            train.add(trainRow);
        }

        br.close();
//        targets.forEach(System.out::println);
//        data.forEach(System.out::println);
        double[][] array = train.stream()
                .map(l -> l.stream().mapToDouble(Double::doubleValue).toArray())
                .toArray(double[][]::new);
        Matrix train_data = Matrix.create(array);
        Vector train_target = Vector.create(targets);
        LinearRegression model = new LinearRegression(0.01,0.1);
        model.fit(train_data, train_target, 1,1);
        }
    }

