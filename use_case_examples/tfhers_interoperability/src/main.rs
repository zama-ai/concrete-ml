use std::env;
use rand::Rng;
use std::path::Path;
use std::fs;
use std::io::Cursor;
use bincode;
use tfhe::set_server_key;
use tfhe::safe_serialization::safe_deserialize;
use std::error::Error;
use tfhe::ServerKey;
use tfhe::prelude::*;
use tfhe::{FheInt8};

const SERIALIZE_SIZE_LIMIT: u64 = 1_000_000_000;
const N_BYTES_IN_AUTH_TOKEN : usize = 16usize;


fn serialize(fheuint: &Vec<FheInt8>, path: &str) {
    let mut serialized_ct = Vec::new();
    bincode::serialize_into(&mut serialized_ct, &fheuint).unwrap();
    let path_ct: &Path = Path::new(path);
    fs::write(path_ct, serialized_ct).unwrap();
}



fn load_serverkey(path: &String) -> Result<ServerKey, Box<dyn Error>>  {
    // Load the server's public evaluation key from a file
    let file = fs::File::open(path)?;
    let data: ServerKey = safe_deserialize(file, SERIALIZE_SIZE_LIMIT)?;
    Ok(data)
}


fn main() -> Result<(), Box<dyn Error>> {

    // Get command-line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: <program> <public_key_path> <input_path> <output_path>");
        std::process::exit(1);
    }

    // Input file paths
    let serverkey_key_path = &args[1];     // e.g., "evalkeys_tfhers.bin"
    let input_path = &args[2];             // e.g., "prediction_non_preprocessed.bin"
    let output_path = &args[3];            // e.g., "tfhers_sign_result.bin"

    // Load and set the server's evaluation key
    let serverkey_key = load_serverkey(&serverkey_key_path)?;
    set_server_key(serverkey_key.clone());


    // Deserialize encrypted input values
    let serialized_list = fs::read(input_path)?;
    let mut serialized_data = Cursor::new(serialized_list);
    let encrypted_values: Vec<FheInt8> = bincode::deserialize_from(&mut serialized_data).unwrap();

    // Generate a random authentication token
    let mut rng = rand::thread_rng();
    let random_tokens: Vec<i8> = (0..N_BYTES_IN_AUTH_TOKEN).map(|_| rng.gen()).collect();
    println!("Random token: {:?}", random_tokens);

    // Compute the argmax: if encrypted_values[0] > encrypted_values[1], select index 0;
    // otherwise index 1
    let diff = &encrypted_values[0] - &encrypted_values[1];
    let sign = diff.gt(0);
    let int_sign = FheInt8::cast_from(sign);

    // Multiply the decision with the random token (homomorphically)
    let response: Vec<FheInt8> = random_tokens.iter().map(|v| (&int_sign * *v)).collect();

    // Serialize the encrypted response
    serialize(&response, output_path);

    Ok(())
}


