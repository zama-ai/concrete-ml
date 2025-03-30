use std::env;
use std::path::Path;
use std::fs;
use std::io::Cursor;
use bincode;
use tfhe::set_server_key;
use tfhe::safe_serialization::safe_deserialize;
use std::error::Error;
use tfhe::ServerKey;
use tfhe::{FheInt8};

const SERIALIZE_SIZE_LIMIT: u64 = 1_000_000_000;


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
    let serverkey_key_path = &args[1];
    let input_path = &args[2];
    let output_path = &args[3];

    // Load and set the server's evaluation key
    let serverkey_key = load_serverkey(&serverkey_key_path)?;
    set_server_key(serverkey_key.clone());

    // Deserialize encrypted input values
    let serialized_list = fs::read(input_path)?;
    let mut serialized_data = Cursor::new(serialized_list);
    let encrypted_values: Vec<FheInt8> = bincode::deserialize_from(&mut serialized_data).unwrap();

    // Mod 2
    let response: Vec<FheInt8> = encrypted_values
    .iter()
    .map(|val| val % 2)
    .collect();

    // Serialize the encrypted response
    serialize(&response, output_path);

    Ok(())
}
