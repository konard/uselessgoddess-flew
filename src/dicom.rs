use {
  dicom::{dictionary_std::tags, object::OpenFileOptions},
  std::{collections::HashMap, path::PathBuf},
  walkdir::WalkDir,
};

#[derive(Debug)]
struct DicomSlice {
  path: PathBuf,
  z: f32,
}

fn scan_dicom_project(root_folder: &str) -> HashMap<String, Vec<DicomSlice>> {
  println!("Scanning DICOM project in: {}", root_folder);

  // Храним снимки, сгруппированные по ID серии (SeriesInstanceUID)
  let mut series_map: HashMap<String, Vec<DicomSlice>> = HashMap::new();

  for entry in WalkDir::new(root_folder).into_iter().filter_map(|e| e.ok()) {
    let path = entry.path();
    if path.is_dir() {
      continue;
    }

    // Пропускаем сам файл оглавления
    if path.file_name().unwrap_or_default() == "DICOMDIR" {
      continue;
    }

    // ОПТИМИЗАЦИЯ: Читаем файл только до тега Pixel Data (без самих картинок)
    // Это ускоряет сканирование тысяч файлов с секунд до миллисекунд
    let obj_result =
      OpenFileOptions::new().read_until(tags::PIXEL_DATA).open_file(path);

    if let Ok(obj) = obj_result {
      // Ищем принадлежность к серии (каждое 3D сканирование имеет уникальный UID)
      let series_uid = obj
        .element_by_name("SeriesInstanceUID")
        .ok()
        .and_then(|e| e.to_str().ok())
        .map(|s| s.into_owned())
        .unwrap_or_else(|| "UnknownSeries".to_string());

      // Ищем Z-координату
      if let Some(z_pos) = obj
        .element_by_name("ImagePositionPatient")
        .ok()
        .and_then(|e| e.to_multi_float32().ok())
        .and_then(|v| v.get(2).copied())
      {
        series_map
          .entry(series_uid)
          .or_default()
          .push(DicomSlice { path: path.to_path_buf(), z: z_pos });
      }
    }
  }

  println!("Found {} distinct volumes/series.", series_map.len());
  series_map
}
