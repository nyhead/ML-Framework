package org.ml;

import java.io.*;
import java.util.*;

public class Main {
    public static void main(String[] args) throws IOException {
        String csvFile = "D:/winter-2023/java_related/ML-Framework/heart.csv";
        BufferedReader br = new BufferedReader(new FileReader(csvFile));
        String line;
        List<Double> targets = new ArrayList<>();
        List<Double> train = new ArrayList<>();
        List<String> data = new ArrayList<>();

        while ((line = br.readLine()) != null) {
            String[] values = line.split(",");
            // Assuming the last column is the target
            for (int i = 0; i < values.length - 1; i++) {
                train.add(Double.valueOf(values[i]));
            }
            targets.add(Double.valueOf(values[values.length - 1]));
            // The rest of the columns are the data
            StringJoiner joiner = new StringJoiner(" ", "", "\n");
            for (int i = 0; i < values.length - 1; i++) {
                joiner.add(values[i]);
            }
            data.add(joiner.toString());
        }

        br.close();
//        targets.forEach(System.out::println);
//        data.forEach(System.out::println);
        }
        // Now you can use targets and datas lists
        // ...
    }

