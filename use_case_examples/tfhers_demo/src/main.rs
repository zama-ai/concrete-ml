use std::path::Path;
use std::fs;
use std::io::Cursor;
use bincode;
use tfhe::core_crypto::prelude::LweSecretKey;
use tfhe::set_server_key;
use tfhe::safe_serialization::safe_deserialize;
use tfhe::FheUint8;
use std::error::Error;
use tfhe::ServerKey;
use tfhe::prelude::*;
use tfhe::{FheInt32};
use std::convert::Into;
use std::fs::File;
use std::io::{Write, BufWriter};

const SERIALIZE_SIZE_LIMIT: u64 = 900_000_000_000_000;

fn serialize(fheuint: &FheInt32, path: &str) {
    let mut serialized_ct = Vec::new();
    bincode::serialize_into(&mut serialized_ct, &fheuint).unwrap();
    let path_ct: &Path = Path::new(path);
    fs::write(path_ct, serialized_ct).unwrap();
}



fn _load_lwe_sk(path: &String) -> Result<LweSecretKey<Vec<u64>>, Box<dyn Error>> {
    let file = fs::File::open(path)?;
    let data = safe_deserialize(file, SERIALIZE_SIZE_LIMIT)?;
    Ok(data)
}

fn load_pk(path: &String) -> Result<ServerKey, Box<dyn Error>>  {
    let file = fs::File::open(path)?;
    let data: ServerKey = safe_deserialize(file, SERIALIZE_SIZE_LIMIT)?;
    Ok(data)
}

fn main() -> Result<(), Box<dyn Error>> {
    let pk_key_path = String::from("serverKey.bin");
    let pk_key = load_pk(&pk_key_path)?;
    set_server_key(pk_key.clone());
    let encrypted_data_path = String::from("encrypted_input.bin");
    let _output_final_score_path = String::from("encrypted_output.bin");

    let path: &Path = Path::new(&encrypted_data_path);
    let serialized_list = fs::read(path).unwrap();
    let mut serialized_data = Cursor::new(serialized_list);

    
    let encrypted_values: Vec<FheUint8> = bincode::deserialize_from(&mut serialized_data).unwrap();

    // let is_conformant = encrypted_values.iter().all(|v| v.is_conformant(&pk_key).into());
    // let is_conformant = encrypted_values.iter().all(|v| v.is_conformant((pk_key).into()));
    // let is_conformant = encrypted_values.iter().all(|v| v.is_conformant((&((&pk).into))));

    let conformance_params = (&pk_key).into();
    let _is_conformant = encrypted_values.iter().all(|v| v.is_conformant(&conformance_params));

    // assert!(is_conformant);

    use rand::Rng;


    let mut rng = rand::thread_rng();
    let random_token: i32 = rng.gen_range(1..=100);
    println!("Random token: {}", random_token);

    let diff = &encrypted_values[0] - &encrypted_values[1];

    let sign = diff.gt(0);
    let int_sign = FheInt32::cast_from(sign);
    let _response = int_sign * random_token;

    Ok(())
}


