use std::fs::File;
use std::io::{BufReader, Write};
use serde_json;
use std::error::Error;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::distributions::Uniform;

fn main() -> Result<(), Box<dyn Error>> {
    let seed: u64 = 727;
    let mut rng = StdRng::seed_from_u64(seed);
    let token: i64 = rng.sample(Uniform::new_inclusive(1, 1000)); 

    println!("Random Token: {}", token);

    let file = File::open("serialized_predict_proba.json")?;
    let reader = BufReader::new(file);
    let y_pred_probas: Vec<Vec<f64>> = serde_json::from_reader(reader)?;

    println!("Desiralised probabilities: {:?}", y_pred_probas);

    let mut argmax_list: Vec<usize> = Vec::new();

    for (sample_index, probas) in y_pred_probas.iter().enumerate() {
        let mut max_index = 0;
        let mut max_value = probas[0];

        for (i, &val) in probas.iter().enumerate() {
            if val > max_value {
                max_value = val;
                max_index = i;
            }
        }

        println!("Slice {}: argmax = {}, value = {}", sample_index, max_index, max_value);

        let result = (max_index as i64) * token;
        println!("        -> Classe {} * token {} = {}", max_index, token, result);

        argmax_list.push(max_index);
    }

    let json_string: String = serde_json::to_string_pretty(&argmax_list)?;
    let mut file = File::create("argmax_results.json")?;
    file.write_all(json_string.as_bytes())?;

    println!("Save results in 'argmax_results.json'");

    Ok(())
}
