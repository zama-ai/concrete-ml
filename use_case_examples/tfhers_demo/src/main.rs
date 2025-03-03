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
use tfhe::{FheInt8};

const SERIALIZE_SIZE_LIMIT: u64 = 900_000_000_000_000;

fn serialize(fheuint: &Vec<FheInt8>, path: &str) {
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
    let pk_key_path = String::from("evalkeys_tfhers.bin");
    let pk_key = load_pk(&pk_key_path)?;
    set_server_key(pk_key.clone());
    let output_final_score_path = String::from("prediction_non_preprocessed.bin");

    let path: &Path = Path::new(&output_final_score_path);
    let serialized_list = fs::read(path).unwrap();
    let mut serialized_data = Cursor::new(serialized_list);

    let encrypted_values: Vec<FheInt8> = bincode::deserialize_from(&mut serialized_data).unwrap();

    // let is_conformant = encrypted_values.iter().all(|v| v.is_conformant(&pk_key).into());
    // let is_conformant = encrypted_values.iter().all(|v| v.is_conformant((pk_key).into()));
    // let is_conformant = encrypted_values.iter().all(|v| v.is_conformant((&((&pk).into))));

//    let is_conformant = encrypted_values.iter().all(|v| v.is_conformant(&(&pk_key).into()));

//    assert!(is_conformant);

    use rand::Rng;

    const N_BYTES_IN_AUTH_TOKEN : usize = 16usize;

    let mut rng = rand::thread_rng();
    let random_tokens: Vec<i8> = (0..N_BYTES_IN_AUTH_TOKEN).map(|_| rng.gen()).collect();
    println!("Random token: {:?}", random_tokens);

    let diff = &encrypted_values[0] - &encrypted_values[1];

    let sign = diff.gt(0);
    let int_sign = FheInt8::cast_from(sign);
    let _response: Vec<FheInt8> = random_tokens.iter().map(|v| (&int_sign * *v)).collect();

    serialize(&_response, "tfhers_sign_result.bin");

    Ok(())
}


